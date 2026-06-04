#!/usr/bin/env bash
set -e


CONTAINER_NAME="eskf_humble"
IMAGE_NAME="eskf-humble:latest"
BASE_OUT="/tmp/linear_turns_delay_no_loss_tdma"

TDMA_SLOT_LENGTHS=(5.0 10.0 20.0 30.0 60.0 120.0)

# echo "🔧 Building Docker image..."
# docker build -t ${IMAGE_NAME} .

echo "🚀 Starting Docker container..."
docker-compose up -d

sleep 3

echo "📊 Running dataset generation inside container..."

docker exec ${CONTAINER_NAME} bash -c '
set -e

source /opt/ros/humble/setup.bash     # system ROS (rclpy etc.)
source /ws/install/setup.bash         # your workspace packages

cd /ws/src/ESKF_rov_tracking

export PYTHONPATH=src:$PYTHONPATH

for SLOT in 5.0 10.0 20.0 30.0 60.0 120.0; do

    # SAFE rate computation
    RATE=$(awk "BEGIN {print 1.0 / ${SLOT}}")

    SLOT_NAME=$(echo ${SLOT} | sed "s/\./p/g")
    OUT_DIR="'"${BASE_OUT}"'_slot_${SLOT_NAME}"

    echo "--------------------------------------------"
    echo "SLOT=${SLOT} RATE=${RATE}"
    echo "OUT=${OUT_DIR}"

    python3 export_fgo_dataset_combined.py \
        --out "${OUT_DIR}" \
        --duration 300 \
        --dt 0.01 \
        --seed 42 \
        --trajectory-type linear_turns \
        --rov-id 1 \
        --epoch-sec 1700000000 \
        --datum-lat 60.3913 \
        --datum-lon 5.3221 \
        --datum-h 0.0 \
        --usbl-rate "${RATE}" \
        --range-rate "${RATE}" \
        --depth-rate "${RATE}" \
        --write-acoustic-rx false \
        --acoustic-delay true \
        --acoustic-jitter-std 0.0 \
        --usbl-miss-prob 0.0 \
        --range-miss-prob 0.0 \
        --depth-miss-prob 0.0 \
        --overwrite

done

echo "✅ Done inside container."
'

echo "📦 Copying datasets..."

mkdir -p ./datasets

for SLOT in 5p0 10p0 20p0 30p0 60p0 120p0; do
    docker cp ${CONTAINER_NAME}:${BASE_OUT}_slot_${SLOT} ./datasets/
done

echo "✅ Done"
