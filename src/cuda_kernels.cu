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

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    int pixelIndex = id * channels;

    unsigned char r = data[pixelIndex];
    unsigned char g = data[pixelIndex + 1];
    unsigned char b = data[pixelIndex + 2];

    unsigned char gray = (r * 0.299 + g * 0.5870 + 0.1140 * b);

    output[id] = gray;
} 


__global__ void flipX_cu(unsigned char *data, int w, int h, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w / 2 && y < h) {
        int left = (x + y*w) * channels;
        int right = ((w - 1- x) + y*w) * channels;

        for (int c=0; c < channels; ++c) {
            unsigned char tmp = data[left + c];
            data[left + c] = data[right + c];
            data[right + c] = tmp;
        }
    }
}

__global__ void flipY_cu(unsigned char *data, int w, int h, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h / 2) {
        int left = (x + y*w) * channels;
        int right = (x + (h - y + 1)*w) * channels;

        for (int c=0; c < channels; ++c) {
            unsigned char tmp = data[left + c];
            data[left + c] = data[right + c];
            data[right + c] = tmp;
        }
    }
}


__global__ void flipYvector_cu(uchar3 *data, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h / 2) {
        int left = x + y * w;
        int right = (x + (h - y + 1) * w);

        uchar3 tmp = data[left];
        data[left] = data[right];
        data[right] = tmp;
    }
}