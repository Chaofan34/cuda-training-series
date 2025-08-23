set -x
filename="$1"
prefix="${filename%.*}"
nvcc -I/root/autodl-tmp/cuda-training-series/exercises ${filename} -o ${prefix}
./${prefix} ${@:2}