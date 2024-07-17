#include "image.h"
#include "opencl_image.h"
#include <cstdlib>
#include <iostream>
#include <chrono>


int main(int argc, char** argv) {
	// Image test("imgs/test.png");
    Image testHD("imgs/testHD.jpeg");
    Image cat("imgs/cat.jpeg");

    Image gpu_test = testHD;

    // std::cout<<cat.channels<<"\n";

    // Mask::GaussianBlur3 gaussianBlur;
    // Mask::EdgeSobelX sobelX;
    // Mask::EdgeSobelY sobelY;

    // Mask::GaussianDynamic1D gaussianBlur1(1, false);
    Mask::GaussianDynamic2D gaussianBlur((int) 1);
    // Mask::GaussianDynamic1D gaussianBlur2(1, true);

    // Timing the computation
    auto start = std::chrono::high_resolution_clock::now();

    cat.std_convolve_clamp_to_0_cpu(0, &gaussianBlur);
    cat.std_convolve_clamp_to_0_cpu(1, &gaussianBlur);
    cat.std_convolve_clamp_to_0_cpu(2, &gaussianBlur);


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;

    cat.write("imgs/tests/2Dgaus3cat0.jpeg");

    
    // // High res
    // start = std::chrono::high_resolution_clock::now();

    // testHD.grayscale_lum_cpu();

    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    // std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;
    // testHD.write("imgs/grayscaleHD.jpg");

    // // Super high res
    // start = std::chrono::high_resolution_clock::now();

    // testB.grayscale_avg_cpu();
    // testB.write("imgs/grayscaleB.jpg");

    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    // std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;

    OpenCLImageProcessor processor;
    // processor.std_convolve_clamp_to_0(gpu_test, &gaussianBlur);
    // processor.std_convolve_clamp_to_0(gpu_test, &gaussianBlur2);
    // processor.diffmap(gpu_cat, test);
    // processor.resizeBicubic(gpu_test, gpu_test.w, gpu_test.h * 1.5);
    // processor.diffmap(gpu_test, testHD);

    // processor.diffmap(gpu_test, gpu_test);

    // gpu_test.write("output/diff.jpeg");

	return 0;
}