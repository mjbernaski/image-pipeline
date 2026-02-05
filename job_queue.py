"""
SQLite-backed persistent job queue for the image pipeline.

Provides thread-safe job queue operations with persistence across restarts.
"""

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class JobStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a generation job."""
    id: str
    status: JobStatus
    payload: Dict[str, Any]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    results: List[Dict] = field(default_factory=list)
    current_run: int = 0
    current_prompt: str = ""
    endpoint_prompts: Dict[str, str] = field(default_factory=dict)
    endpoint_status: Dict[str, Dict] = field(default_factory=dict)
    llm_status: Dict[str, Any] = field(default_factory=dict)
    total_elapsed: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for API responses."""
        return {
            "id": self.id,
            "status": self.status.value,
            "prompt": self.payload.get("prompt", ""),
            "prompt2": self.payload.get("prompt2", ""),
            "prompt_mode": self.payload.get("prompt_mode", "same"),
            "random": self.payload.get("use_random", False),
            "count": self.payload.get("count", 1),
            "has_image": self.payload.get("image_base64") is not None,
            "current_run": self.current_run,
            "current_prompt": self.current_prompt,
            "endpoint_prompts": self.endpoint_prompts,
            "endpoint_status": self.endpoint_status,
            "llm_status": self.llm_status,
            "results": self.results,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_elapsed": self.total_elapsed,
            "error": self.error,
        }


class PersistentJobQueue:
    """
    Thread-safe SQLite-backed job queue with persistence.

    Jobs survive server restarts. Interrupted jobs (status=RUNNING when
    server starts) are automatically recovered to PENDING state.
    """

    def __init__(self, db_path: str = "jobs.db"):
        """
        Initialize the persistent job queue.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._shutdown = False
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    results TEXT DEFAULT '[]',
                    current_run INTEGER DEFAULT 0,
                    current_prompt TEXT DEFAULT '',
                    endpoint_prompts TEXT DEFAULT '{}',
                    endpoint_status TEXT DEFAULT '{}',
                    llm_status TEXT DEFAULT '{}',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    total_elapsed REAL,
                    queue_position INTEGER
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)
            """)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        """Convert a database row to a Job object."""
        return Job(
            id=row["id"],
            status=JobStatus(row["status"]),
            payload=json.loads(row["payload"]),
            results=json.loads(row["results"]),
            current_run=row["current_run"],
            current_prompt=row["current_prompt"],
            endpoint_prompts=json.loads(row["endpoint_prompts"]),
            endpoint_status=json.loads(row["endpoint_status"]),
            llm_status=json.loads(row["llm_status"]),
            error=row["error"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            total_elapsed=row["total_elapsed"],
        )

    def enqueue(self, job_id: str, payload: Dict[str, Any]) -> Job:
        """
        Add a new job to the queue.

        Args:
            job_id: Unique job identifier
            payload: Job parameters (prompt, count, etc.)

        Returns:
            The created Job object
        """
        with self._lock:
            created_at = datetime.now().isoformat()

            with self._get_connection() as conn:
                pending_count = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN (?, ?)",
                    (JobStatus.QUEUED.value, JobStatus.RUNNING.value)
                ).fetchone()[0]

                conn.execute("""
                    INSERT INTO jobs (id, status, payload, created_at, queue_position)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    job_id,
                    JobStatus.QUEUED.value,
                    json.dumps(payload),
                    created_at,
                    pending_count
                ))
                conn.commit()

            job = Job(
                id=job_id,
                status=JobStatus.QUEUED,
                payload=payload,
                created_at=created_at,
            )

            self._condition.notify_all()
            return job

    def dequeue(self, timeout: Optional[float] = None) -> Optional[Job]:
        """
        Get the next pending job from the queue.

        Blocks until a job is available or timeout expires.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            Job object or None if timeout/shutdown
        """
        with self._condition:
            deadline = time.time() + timeout if timeout else None

            while not self._shutdown:
                with self._get_connection() as conn:
                    row = conn.execute("""
                        SELECT * FROM jobs
                        WHERE status = ?
                        ORDER BY created_at ASC
                        LIMIT 1
                    """, (JobStatus.QUEUED.value,)).fetchone()

                    if row:
                        job = self._row_to_job(row)
                        conn.execute("""
                            UPDATE jobs SET status = ?, started_at = ?
                            WHERE id = ?
                        """, (JobStatus.RUNNING.value, datetime.now().isoformat(), job.id))
                        conn.commit()
                        job.status = JobStatus.RUNNING
                        job.started_at = datetime.now().isoformat()
                        return job

                if deadline:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return None
                    self._condition.wait(timeout=remaining)
                else:
                    self._condition.wait(timeout=1.0)

            return None

    def get(self, job_id: str) -> Optional[Job]:
        """
        Get a job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job object or None if not found
        """
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()
                return self._row_to_job(row) if row else None

    def update(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        results: Optional[List[Dict]] = None,
        current_run: Optional[int] = None,
        current_prompt: Optional[str] = None,
        endpoint_prompts: Optional[Dict[str, str]] = None,
        endpoint_status: Optional[Dict[str, Dict]] = None,
        llm_status: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        total_elapsed: Optional[float] = None,
    ) -> Optional[Job]:
        """
        Update a job's state.

        Args:
            job_id: Job identifier
            status: New status
            results: Generation results
            current_run: Current run number
            current_prompt: Current prompt being processed
            endpoint_prompts: Prompts for each endpoint
            endpoint_status: Status for each endpoint
            llm_status: LLM generation status
            error: Error message if failed
            total_elapsed: Total time elapsed

        Returns:
            Updated Job object or None if not found
        """
        with self._lock:
            with self._get_connection() as conn:
                updates = []
                values = []

                if status is not None:
                    updates.append("status = ?")
                    values.append(status.value)
                    if status == JobStatus.COMPLETE or status == JobStatus.ERROR:
                        updates.append("completed_at = ?")
                        values.append(datetime.now().isoformat())

                if results is not None:
                    updates.append("results = ?")
                    values.append(json.dumps(results))

                if current_run is not None:
                    updates.append("current_run = ?")
                    values.append(current_run)

                if current_prompt is not None:
                    updates.append("current_prompt = ?")
                    values.append(current_prompt)

                if endpoint_prompts is not None:
                    updates.append("endpoint_prompts = ?")
                    values.append(json.dumps(endpoint_prompts))

                if endpoint_status is not None:
                    updates.append("endpoint_status = ?")
                    values.append(json.dumps(endpoint_status))

                if llm_status is not None:
                    updates.append("llm_status = ?")
                    values.append(json.dumps(llm_status))

                if error is not None:
                    updates.append("error = ?")
                    values.append(error)

                if total_elapsed is not None:
                    updates.append("total_elapsed = ?")
                    values.append(total_elapsed)

                if not updates:
                    return self.get(job_id)

                values.append(job_id)
                conn.execute(
                    f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?",
                    values
                )
                conn.commit()

            return self.get(job_id)

    def complete(self, job_id: str, results: List[Dict], total_elapsed: float) -> Optional[Job]:
        """
        Mark a job as complete.

        Args:
            job_id: Job identifier
            results: Generation results
            total_elapsed: Total time taken

        Returns:
            Updated Job object
        """
        return self.update(
            job_id,
            status=JobStatus.COMPLETE,
            results=results,
            total_elapsed=total_elapsed,
        )

    def fail(self, job_id: str, error: str) -> Optional[Job]:
        """
        Mark a job as failed.

        Args:
            job_id: Job identifier
            error: Error message

        Returns:
            Updated Job object
        """
        return self.update(
            job_id,
            status=JobStatus.ERROR,
            error=error,
        )

    def cancel(self, job_id: str) -> Optional[Job]:
        """
        Cancel a queued job.

        Args:
            job_id: Job identifier

        Returns:
            Updated Job object or None if not found/not cancellable
        """
        job = self.get(job_id)
        if job and job.status == JobStatus.QUEUED:
            return self.update(job_id, status=JobStatus.CANCELLED)
        return None

    def recover_interrupted(self) -> List[Job]:
        """
        Recover jobs that were running when the server stopped.

        Moves RUNNING jobs back to QUEUED status so they can be retried.

        Returns:
            List of recovered jobs
        """
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE status = ?",
                    (JobStatus.RUNNING.value,)
                ).fetchall()

                recovered = []
                for row in rows:
                    conn.execute(
                        "UPDATE jobs SET status = ?, started_at = NULL WHERE id = ?",
                        (JobStatus.QUEUED.value, row["id"])
                    )
                    job = self._row_to_job(row)
                    job.status = JobStatus.QUEUED
                    job.started_at = None
                    recovered.append(job)

                conn.commit()
                return recovered

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status.

        Returns:
            Dict with pending, running, and completed counts
        """
        with self._lock:
            with self._get_connection() as conn:
                result = conn.execute("""
                    SELECT status, COUNT(*) as count
                    FROM jobs
                    GROUP BY status
                """).fetchall()

                counts = {row["status"]: row["count"] for row in result}

                running_job = conn.execute(
                    "SELECT id FROM jobs WHERE status = ? LIMIT 1",
                    (JobStatus.RUNNING.value,)
                ).fetchone()

                return {
                    "pending": counts.get(JobStatus.QUEUED.value, 0),
                    "running": counts.get(JobStatus.RUNNING.value, 0),
                    "completed": counts.get(JobStatus.COMPLETE.value, 0),
                    "failed": counts.get(JobStatus.ERROR.value, 0),
                    "cancelled": counts.get(JobStatus.CANCELLED.value, 0),
                    "current_job_id": running_job["id"] if running_job else None,
                }

    def get_all_jobs(self, limit: int = 100) -> Dict[str, List[Job]]:
        """
        Get all jobs grouped by status.

        Args:
            limit: Maximum number of completed jobs to return

        Returns:
            Dict with pending, running, and completed lists
        """
        with self._lock:
            with self._get_connection() as conn:
                pending = [
                    self._row_to_job(row) for row in conn.execute(
                        "SELECT * FROM jobs WHERE status = ? ORDER BY created_at ASC",
                        (JobStatus.QUEUED.value,)
                    ).fetchall()
                ]

                running = [
                    self._row_to_job(row) for row in conn.execute(
                        "SELECT * FROM jobs WHERE status = ? ORDER BY started_at ASC",
                        (JobStatus.RUNNING.value,)
                    ).fetchall()
                ]

                completed = [
                    self._row_to_job(row) for row in conn.execute(
                        """SELECT * FROM jobs
                           WHERE status IN (?, ?, ?)
                           ORDER BY completed_at DESC LIMIT ?""",
                        (JobStatus.COMPLETE.value, JobStatus.ERROR.value, JobStatus.CANCELLED.value, limit)
                    ).fetchall()
                ]

                return {
                    "pending": pending,
                    "running": running,
                    "completed": completed,
                }

    def clear_queue(self) -> int:
        """
        Clear all queued (pending) jobs.

        Returns:
            Number of jobs cleared
        """
        with self._lock:
            with self._get_connection() as conn:
                result = conn.execute(
                    "DELETE FROM jobs WHERE status = ?",
                    (JobStatus.QUEUED.value,)
                )
                count = result.rowcount
                conn.commit()
                return count

    def size(self) -> int:
        """Get number of pending jobs in queue."""
        with self._lock:
            with self._get_connection() as conn:
                result = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status = ?",
                    (JobStatus.QUEUED.value,)
                ).fetchone()
                return result[0]

    def shutdown(self) -> None:
        """Signal the queue to shut down gracefully."""
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()


class JobState:
    """
    Thread-safe container for in-memory job state during generation.

    Used for real-time status updates that don't need persistence.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._current_job_id: Optional[str] = None
        self._active_jobs: Dict[str, Dict[str, Any]] = {}

    @property
    def current_job_id(self) -> Optional[str]:
        """Get the currently running job ID."""
        with self._lock:
            return self._current_job_id

    @current_job_id.setter
    def current_job_id(self, value: Optional[str]) -> None:
        """Set the currently running job ID."""
        with self._lock:
            self._current_job_id = value

    def set_job_state(self, job_id: str, key: str, value: Any) -> None:
        """Set a state value for a job."""
        with self._lock:
            if job_id not in self._active_jobs:
                self._active_jobs[job_id] = {}
            self._active_jobs[job_id][key] = value

    def get_job_state(self, job_id: str, key: str, default: Any = None) -> Any:
        """Get a state value for a job."""
        with self._lock:
            return self._active_jobs.get(job_id, {}).get(key, default)

    def clear_job_state(self, job_id: str) -> None:
        """Clear all state for a job."""
        with self._lock:
            self._active_jobs.pop(job_id, None)
