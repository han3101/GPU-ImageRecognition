#pragma once
#include <cuda_runtime.h>


__global__ void grayscale_avg_cu(const unsigned char *data, unsigned char *output, int channels);

__global__ void grayscale_lum_cu(const unsigned char *data, unsigned char *output, int channels);

__global__ void flipX_cu(unsigned char *data, int w, int h, int channels);

__global__ void flipY_cu(unsigned char *data, int w, int h, int channels);

__global__ void flipYvector_cu(uchar3 *data, int w, int h);