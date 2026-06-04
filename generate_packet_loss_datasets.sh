#!/usr/bin/env bash
set -e


CONTAINER_NAME="eskf_humble"
IMAGE_NAME="eskf-humble:latest"
BASE_OUT="/tmp/linear_turns_delay_tdma_loss"

PACKET_LOSS_PROB=(0.0 0.1 0.3 0.5 0.7 0.9)
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

for LOSS in 0.0 0.1 0.3 0.5 0.7 0.9; do

    # SAFE rate computation
    RATE=$(awk "BEGIN {print 1.0 / ${LOSS}}")

    LOSS_NAME=$(echo ${LOSS} | sed "s/\./p/g")
    OUT_DIR="'"${BASE_OUT}"'_loss_${LOSS_NAME}"

    echo "--------------------------------------------"
    echo "LOSS=${LOSS} RATE=${RATE}"
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
        --usbl-rate 0.2 \
        --range-rate 0.2 \
        --depth-rate 0.2 \
        --write-acoustic-rx false \
        --acoustic-delay true \
        --acoustic-jitter-std 0.0 \
        --usbl-miss-prob ${LOSS} \
        --range-miss-prob ${LOSS} \
        --depth-miss-prob ${LOSS} \
        --overwrite

done

echo "✅ Done inside container."
'

echo "📦 Copying datasets..."

mkdir -p ./datasets

for LOSS in 0p0 0p1 0p3 0p5 0p7 0p9; do
    docker cp ${CONTAINER_NAME}:${BASE_OUT}_loss_${LOSS} ./datasets/
done

echo "✅ Done"
