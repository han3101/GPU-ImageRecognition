#pragma once
#include <cuda_runtime.h>


__global__ void grayscale_avg_cu(const unsigned char *data, unsigned char *output, int channels);

__global__ void grayscale_lum_cu(const unsigned char *data, unsigned char *output, int channels);