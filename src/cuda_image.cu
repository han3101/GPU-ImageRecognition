#include "cuda_image.cuh"
#include <iostream>
#include <cstdlib>
#include <omp.h>

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


void CUDAImageProcessor::flipX(Image& image) {

    // Allocate memory buffers
    size_t bytes_i = image.size * sizeof(uint8_t);

    uint8_t *data_d;
    cudaMalloc(&data_d, bytes_i);

    cudaMemcpy(data_d, image.data, bytes_i, cudaMemcpyHostToDevice);

    int GRID_X = (image.w + THREADS - 1) / THREADS;
    int GRID_Y = (image.h + THREADS - 1) / THREADS;

    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(GRID_X, GRID_Y);

    flipX_cu<<<grid_dim, block_dim>>>(data_d, image.w, image.h, image.channels);
    cudaDeviceSynchronize();

    cudaMemcpy(image.data, data_d, bytes_i, cudaMemcpyDeviceToHost);

    cudaFree(data_d);
}

void CUDAImageProcessor::flipY(Image& image) {

    // Allocate memory buffers
    size_t bytes_i = image.size * sizeof(uint8_t);

    uint8_t *data_d;
    cudaMalloc(&data_d, bytes_i);

    cudaMemcpy(data_d, image.data, bytes_i, cudaMemcpyHostToDevice);

    int GRID_X = (image.w + THREADS - 1) / THREADS;
    int GRID_Y = (image.h + THREADS - 1) / THREADS;

    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(GRID_X, GRID_Y);

    flipY_cu<<<grid_dim, block_dim>>>(data_d, image.w, image.h, image.channels);
    cudaDeviceSynchronize();

    cudaMemcpy(image.data, data_d, bytes_i, cudaMemcpyDeviceToHost);

    cudaFree(data_d);
}

void CUDAImageProcessor::flipYvector(Image& image) {

    if (image.channels != 3) {
        std::cout<<"flipYvector only for 3 channel images, using flipY"<<std::endl;
        this->flipY(image);
        return;
    }

    // Allocate memory buffers
    std::vector<uchar3> data_h(image.w * image.h);

    #pragma omp parallel for
    for (int i=0; i < image.w * image.h; i++) {
        data_h[i].x = image.data[i * image.channels];
        data_h[i].y = image.data[i * image.channels + 1];
        data_h[i].z = image.data[i * image.channels + 2];
    }
    

    uchar3* data_d;
    cudaMalloc(&data_d, image.w * image.h * sizeof(uchar3));

    cudaMemcpy(data_d, data_h.data(), image.w * image.h * sizeof(uchar3), cudaMemcpyHostToDevice);

    int GRID_X = (image.w + THREADS - 1) / THREADS;
    int GRID_Y = (image.h + THREADS - 1) / THREADS;

    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(GRID_X, GRID_Y);

    flipYvector_cu<<<grid_dim, block_dim>>>(data_d, image.w, image.h);
    cudaDeviceSynchronize();

    cudaMemcpy(data_h.data(), data_d, image.w * image.h * sizeof(uchar3), cudaMemcpyDeviceToHost);


    #pragma omp parallel for
    for (int i=0; i < image.w * image.h; i++) {
        image.data[i * image.channels] = data_h[i].x;
        image.data[i * image.channels + 1] = data_h[i].y;
        image.data[i * image.channels + 2] = data_h[i].z;
    }

    cudaFree(data_d);
}

void CUDAImageProcessor::resizeBilinear(Image& image, int nw, int nh) {

    // Allocate memory buffers
    size_t bytes_i = image.size * sizeof(uint8_t);
    size_t bytes_o = nw * nh * image.channels * sizeof(uint8_t);


    uint8_t *data_d, *output_d;
    cudaMalloc(&data_d, bytes_i);
    cudaMalloc(&output_d, bytes_o);

    cudaMemcpy(data_d, image.data, bytes_i, cudaMemcpyHostToDevice);

    int GRID_X = (nw + THREADS - 1) / THREADS;
    int GRID_Y = (nh + THREADS - 1) / THREADS;

    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(GRID_X, GRID_Y);

    float scaleX = (float) (image.w-1) / (nw-1);
    float scaleY = (float) (image.h-1) / (nh-1);

    resize_bilinear_cu<<<grid_dim, block_dim>>>(data_d, output_d, nw, nh, image.w, image.h, image.channels, scaleX, scaleY);
    cudaDeviceSynchronize();

    image.size = nw * nh * image.channels;
	uint8_t* newData = new uint8_t[image.size];
    image.w = nw;
	image.h = nh;
	delete[] image.data;
	image.data = newData;
	newData = nullptr;

    cudaMemcpy(image.data, output_d, bytes_o, cudaMemcpyDeviceToHost);

    cudaFree(data_d);
    cudaFree(output_d);

}