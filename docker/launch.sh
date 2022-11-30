cd ..
CUDA_VISIBLE_DEVICES='7'

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --name $1 \
    --mount src=$(pwd),dst=/faster-rcnn-inference,type=bind \
    --mount src=/media/data,dst=/data,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -p 8888:8888 \
    -w /faster-rcnn-inference \
    litcoderr/frcnn:latest \
    bash -c "bash" \
