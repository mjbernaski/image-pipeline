#!/bin/bash

cd "$(dirname "$0")"

kill_port() {
    local port=$1
    local pid=$(lsof -ti :$port 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "  Killing existing process on port $port (PID: $pid)"
        kill -TERM $pid 2>/dev/null
        sleep 1
        kill -9 $pid 2>/dev/null
    fi
}

echo "Checking for existing processes on ports..."
kill_port 5050
kill_port 4000
kill_port 3030
echo ""

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

PID1=""
PID2=""
PID3=""

cleanup() {
    echo ""
    echo "Initiating graceful shutdown..."

    if [ -n "$PID1" ]; then
        echo "  Stopping Dual Gen Server (PID: $PID1)..."
        kill -TERM $PID1 2>/dev/null
    fi

    if [ -n "$PID2" ]; then
        echo "  Stopping JSON from Image Server (PID: $PID2)..."
        kill -TERM $PID2 2>/dev/null
    fi

    if [ -n "$PID3" ]; then
        echo "  Stopping Create JSON Server (PID: $PID3)..."
        kill -TERM $PID3 2>/dev/null
    fi

    echo "Waiting for processes to finish..."
    wait $PID1 2>/dev/null
    wait $PID2 2>/dev/null
    wait $PID3 2>/dev/null

    echo "All services stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

check_health() {
    local port=$1
    local name=$2
    local max_attempts=10
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "  [OK] $name is healthy"
            return 0
        fi
        sleep 0.5
        attempt=$((attempt + 1))
    done

    echo "  [WARN] $name health check failed (may still be starting)"
    return 1
}

echo "Starting Image Pipeline Services..."
echo ""

python dual_gen_server.py &
PID1=$!
echo "  [1] Dual Gen Server (PID: $PID1) - Port 5050"

sleep 2

python json_from_image_server.py &
PID2=$!
echo "  [2] JSON from Image Server (PID: $PID2) - Port 4000"

sleep 1

python create_json_server.py &
PID3=$!
echo "  [3] Create JSON Server (PID: $PID3) - Port 3030"

echo ""
echo "Checking service health..."

sleep 2
check_health 5050 "Dual Gen Server"
check_health 4000 "JSON from Image Server"
check_health 3030 "Create JSON Server"

echo ""
echo "All services running. Press Ctrl+C to stop all."
echo ""
echo "Endpoints:"
echo "  - Dual Gen UI:        http://localhost:5050"
echo "  - Dual Gen API (v1):  http://localhost:5050/api/v1/"
echo "  - JSON from Image:    http://localhost:4000"
echo "  - Create JSON:        http://localhost:3030"
echo ""
echo "Health checks:"
echo "  - curl http://localhost:5050/health"
echo "  - curl http://localhost:4000/health"
echo "  - curl http://localhost:3030/health"
echo ""

wait
