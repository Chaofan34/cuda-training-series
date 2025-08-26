set -x
filename="$1"
prefix="${filename%.*}"
nvcc -arch=sm_70 -I/root/autodl-tmp/cuda-training-series/exercises -lcublas ${filename} -o ${prefix}
shift
./${prefix} $@
