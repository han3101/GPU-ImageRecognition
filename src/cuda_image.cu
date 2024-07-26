#include "cuda_image.cuh"
#include <iostream>
#include <cstdlib>

CUDAImageProcessor::CUDAImageProcessor() {}

CUDAImageProcessor::~CUDAImageProcessor() {}


void CUDAImageProcessor::grayscale_avg(Image& image) {

    // Allocate memory buffers
    size_t bytes_n = image.w * image.h * sizeof(uint8_t);
    size_t bytes_i = image.size * sizeof(uint8_t);
    uint8_t* newData = new uint8_t[bytes_n];

    uint8_t *data_d, *result_d;
    cudaMalloc(&result_d, bytes_n);
    cudaMalloc(&data_d, bytes_i);

    cudaMemcpy(data_d, image.data, bytes_i, cudaMemcpyHostToDevice);

    int GRID = (bytes_n + THREADS - 1) / THREADS;

    grayscale_avg_cu<<<GRID, THREADS>>>(data_d, result_d, image.channels);
    cudaDeviceSynchronize();

    cudaMemcpy(newData, result_d, bytes_n, cudaMemcpyDeviceToHost);

    delete[] image.data;
    image.data = newData;
    image.size = bytes_n;
    image.channels = 1;

    cudaFree(result_d);
    cudaFree(data_d);


}

void CUDAImageProcessor::grayscale_lum(Image& image) {

    // Allocate memory buffers
    size_t bytes_n = image.w * image.h * sizeof(uint8_t);
    size_t bytes_i = image.size * sizeof(uint8_t);
    uint8_t* newData = new uint8_t[bytes_n];

    uint8_t *data_d, *result_d;
    cudaMalloc(&result_d, bytes_n);
    cudaMalloc(&data_d, bytes_i);

    cudaMemcpy(data_d, image.data, bytes_i, cudaMemcpyHostToDevice);

    int GRID = (bytes_n + THREADS - 1) / THREADS;

    grayscale_lum_cu<<<GRID, THREADS>>>(data_d, result_d, image.channels);
    cudaDeviceSynchronize();

    cudaMemcpy(newData, result_d, bytes_n, cudaMemcpyDeviceToHost);

    delete[] image.data;
    image.data = newData;
    image.size = bytes_n;
    image.channels = 1;

    cudaFree(result_d);
    cudaFree(data_d);
}
