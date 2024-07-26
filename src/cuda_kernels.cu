#include "cuda_image.cuh"

/*
* For optimization use with cudaMemcpyToSymbol
* Constant image size might need to be bigger
*/
// __constant__ unsigned char image_data[1024 * 1024];

__global__ void grayscale_avg_cu(const unsigned char *data, unsigned char *output, int channels) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    int pixelIndex = id * channels;

    unsigned char r = data[pixelIndex];
    unsigned char g = data[pixelIndex + 1];
    unsigned char b = data[pixelIndex + 2];

    unsigned char gray = (r + g + b) / 3;

    output[id] = gray;

} 

__global__ void grayscale_lum_cu(const unsigned char *data, unsigned char *output, int channels) {

    int id = blockIdx.x * blockDim.x  + threadIdx.x;

    int pixelIndex = id * channels;

    unsigned char r = data[pixelIndex];
    unsigned char g = data[pixelIndex + 1];
    unsigned char b = data[pixelIndex + 2];

    unsigned char gray = (r * 0.299 + g * 0.5870 + 0.1140 * b);

    output[id] = gray;
} 