#pragma once

#define CL_TARGET_OPENCL_VERSION 220
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl2.hpp>
#include "image.h"
#include <stdint.h>
// #include "PNG.h"

class OpenCLImageProcessor {
public:
    OpenCLImageProcessor();
    ~OpenCLImageProcessor();

    void init();
    void grayscale_avg(Image& image);
    void grayscale_lum(Image& image);

    void diffmap(Image& image1, Image& image2);

    void flipX(Image& image);
    void flipY(Image& image);

    void resizeBilinear(Image& image, int nw, int nh);
    void resizeBicubic(Image& image, int nw, int nh);

    void std_convolve_clamp_to_0(Image& image, const Mask::BaseMask* mask);
    void std_convolve_clamp_to_border(Image& image, const Mask::BaseMask* mask);
    void std_convolve_clamp_to_cyclic(Image& image, const Mask::BaseMask* mask);

    void local_binary_pattern(Image& image);
    void integralImage(Image& image, std::unique_ptr<u_int32_t[]>& integralImage, std::unique_ptr<u_int32_t[]>& integralImageSquare, std::unique_ptr<u_int32_t[]>& integralImageTilt,  std::unique_ptr<u_int32_t[]>& integralImageSobel);
    void evalStages(Image& image, std::vector<double>& haar, std::vector<int>& results, std::unique_ptr<u_int32_t[]>& integralImage, std::unique_ptr<u_int32_t[]>& integralImageSquare, std::unique_ptr<u_int32_t[]>& integralImageTilt, std::unique_ptr<u_int32_t[]>& integralImageSobel, int blockWidth, int blockHeight, float scale, float inverseArea, int step, float edgeDensity);

private:
    cl::Context m_context;
    cl::Device m_device;
    // cl::Program program;
    cl::CommandQueue m_queue;

    void loadKernels();
    std::string loadKernelSource(const std::string& fileName);
    
    std::string getErrorString(cl_int error);
};