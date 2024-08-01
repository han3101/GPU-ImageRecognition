#include "image.h"
#if USE_OPENCL
#include "opencl_image.h"
#endif
#include "viola_jones.h"
#if USE_CUDA
#include "cuda_image.cuh"
#endif
#include <cstdlib>
#include <iostream>
#include <chrono>


int main(int argc, char** argv) {
	// Image test("imgs/test.png");
    // Image testHD("imgs/testHD.jpeg");
    // Image cat("imgs/cat.jpeg");
    Image tkl("imgs/tkl.jpg");

    // Image gpu_test = cat;

    // std::cout<<cat.channels<<"\n";

    // Mask::GaussianBlur3 gaussianBlur;
    Mask::EdgeSobelX sobelX;
    Mask::EdgeSobelY sobelY;

    // Mask::GaussianDynamic1D gaussianBlur1(1, false);
    // Mask::GaussianDynamic2D gaussianBlur((int) 1);
    // Mask::GaussianDynamic1D gaussianBlur2(1, true);

    // Timing the computation
    auto start = std::chrono::high_resolution_clock::now();

    // cat.grayscale_avg_lum();
    // cat.local_binary_pattern_cpu();
    // cat.std_convolve_clamp_to_0_cpu(0, &sobelX);
    // cat.std_convolve_clamp_to_0_cpu(1, &sobelX);
    // cat.std_convolve_clamp_to_0_cpu(2, &sobelX);

    // cat.std_convolve_clamp_to_0_cpu(0, &sobelY);
    // cat.std_convolve_clamp_to_0_cpu(1, &sobelY);
    // cat.std_convolve_clamp_to_0_cpu(2, &sobelY);

    ViolaJones faceTrack;
    OpenCLImageProcessor processor;
    CUDAImageProcessor cudap;
    HaarCasscades haar;
    faceTrack.set_stepSize(1.5);

    Image colortkl = tkl;
    // std::vector<Rect> faces = faceTrack.detect(tkl, haar.haar_face);
    // std::vector<Rect> faces = faceTrack.detect(tkl, haar.haar_face, processor);
    std::vector<Rect> faces = faceTrack.detect(tkl, haar.haar_face, cudap);
    
    std::cout<<"Before draw"<<"\n";
    faceTrack.draw(colortkl, faces);


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;

    colortkl.write("output/lbp.jpeg");


    // OpenCLImageProcessor processor;
    // processor.std_convolve_clamp_to_0(cat, &sobelX);
    // processor.std_convolve_clamp_to_0(tkl, &sobelY);
    // // processor.diffmap(gpu_cat, test);
    // // processor.resizeBicubic(gpu_test, gpu_test.w, gpu_test.h * 1.5);
    // // processor.diffmap(gpu_test, testHD);

    // // processor.diffmap(gpu_test, gpu_test);
    // // processor.local_binary_pattern(cat);

    // processor.flipX(tkl);

    // CUDAImageProcessor cudap;

    // cudap.std_convolve_clamp_to_0(gpu_test, &sobelX);

    // gpu_test.write("output/cat_gpu.jpeg");

    // cat.diffmap_cpu(gpu_test);
    // cudap.resizeBilinear(tkl, 2000, 2000);
    // // processor.resizeBilinear(tkl, 2000, 2000);

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;
    // cat.write("output/diff.jpeg");

	return 0;
}