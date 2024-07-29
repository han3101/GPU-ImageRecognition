#pragma once

#include "image.h"
// #include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

class CUDAImageProcessor
{
public:
    CUDAImageProcessor();
    ~CUDAImageProcessor();

    void grayscale_avg(Image &image);
    void grayscale_lum(Image &image);

    void flipX(Image &image);
    void flipY(Image &image);
    void flipYvector(Image &image);

    void resizeBilinear(Image& image, int nw, int nh);

    void std_convolve_clamp_to_0(Image &image, const Mask::BaseMask *mask);
    void std_convolve_clamp_to_border(Image &image, const Mask::BaseMask *mask);

    void integralImage(Image& image, std::unique_ptr<u_int32_t[]>& integralImage, std::unique_ptr<u_int32_t[]>& integralImageSquare, std::unique_ptr<u_int32_t[]>& integralImageTilt,  std::unique_ptr<u_int32_t[]>& integralImageSobel);
    void evalStages(Image& image, std::vector<double>& haar, std::vector<int>& results, std::unique_ptr<u_int32_t[]>& integralImage, std::unique_ptr<u_int32_t[]>& integralImageSquare, std::unique_ptr<u_int32_t[]>& integralImageTilt, std::unique_ptr<u_int32_t[]>& integralImageSobel, int blockWidth, int blockHeight, float scale, float inverseArea, int step, float edgeDensity);

private:
    int THREADS = 32;
};


__global__ void grayscale_avg_cu(const unsigned char *data, unsigned char *output, int channels);

__global__ void grayscale_lum_cu(const unsigned char *data, unsigned char *output, int channels);

__global__ void flipX_cu(unsigned char *data, int w, int h, int channels);

__global__ void flipY_cu(unsigned char *data, int w, int h, int channels);

__global__ void flipYvector_cu(uchar3 *data, int w, int h);

__global__ void resize_bilinear_cu(unsigned char *data, unsigned char *output, int nw, int nh, int w, int h, int channels, float scaleX, float scaleY);