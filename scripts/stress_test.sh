#!/bin/bash

# Stress test script using wrk
# Usage: ./stress_test.sh <API_URL> <CONNECTIONS> <DURATION> <THREADS>

API_URL=$1
CONNECTIONS=${2:-1000}
DURATION=${3:-30s}
THREADS=${4:-10}

echo "=========================================="
echo "Starting Stress Test"
echo "=========================================="
echo "API URL: $API_URL"
echo "Connections: $CONNECTIONS"
echo "Duration: $DURATION"
echo "Threads: $THREADS"
echo "=========================================="

# Create a Lua script for POST requests
cat > /tmp/post.lua << 'EOF'
wrk.method = "POST"
wrk.body   = '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
wrk.headers["Content-Type"] = "application/json"
EOF

# Run wrk
wrk -t${THREADS} -c${CONNECTIONS} -d${DURATION} -s /tmp/post.lua ${API_URL}/predict

# Cleanup
rm -f /tmp/post.lua

echo ""
echo "=========================================="
echo "Stress Test Completed"
echo "=========================================="
