set -x
filename="$1"
prefix="${filename%.*}"
nvcc -arch=sm_70 -I/root/autodl-tmp/cuda-training-series/exercises -G -g -std=c++14 ${filename} -o ${prefix} -lineinfo 
shift
./${prefix} $@
