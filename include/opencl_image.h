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

private:
    cl::Context m_context;
    cl::Device m_device;
    // cl::Program program;
    cl::CommandQueue m_queue;

    void loadKernels();
    std::string loadKernelSource(const std::string& fileName);
    
    std::string getErrorString(cl_int error);
};