#include <gtest/gtest.h>

#include "image.h"
#include "opencl_image.h"
#ifdef USE_CUDA
#include "cuda_image.cuh"
#endif
#include "masks.h"
#include <cstdlib>
#include <iostream>
#include <memory>

int is_image_black(const Image& img) {
    int isBlack = 1;

    #pragma omp parallel for num_threads(4) shared(isBlack)
    for (int i = 0; i < img.size; ++i) {
        if (img.data[i] > 5) {
            #pragma omp critical
            {
                isBlack = 0;
                // std::cout<<"Pixel: "<<i<<" has value "<<(int)img.data[i]<<"\n";
            }
        }
    }

    return isBlack;
}

int is_image_black_single(const Image& img) {
    for (int i = 0; i < img.size; ++i) {
        if (img.data[i] > 5) {
            {
                return 0;
            }
        }
    }

    return 1;
}

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

TEST(ImageTest, BasicDiffTest) {

    Image testHD("imgs/testHD.jpeg");

    ASSERT_NE(testHD.data, nullptr) << "Failed to load image.";


    OpenCLImageProcessor processor;

    processor.diffmap(testHD, testHD);

    auto start = std::chrono::high_resolution_clock::now();

    int is_black = is_image_black(testHD);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;


    EXPECT_EQ(is_black, 1);
}

// TEST(ImageTest, 2DDynamicGaus3) {

//     Image testHD("imgs/cat.jpeg");
//     Image target("imgs/tests/2Dgaus3cat.jpeg");

//     ASSERT_NE(testHD.data, nullptr) << "Failed to load test image.";
//     ASSERT_NE(target.data, nullptr) << "Failed to load target image.";

//     Mask::GaussianDynamic2D gaussianBlur((int) 1);

//     OpenCLImageProcessor processor;
//     processor.std_convolve_clamp_to_border(testHD, &gaussianBlur);

//     processor.diffmap(testHD, target);

//     auto start = std::chrono::high_resolution_clock::now();

//     int is_black = is_image_black(testHD);

//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end - start;
//     std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;


//     EXPECT_EQ(is_black, 1);
// }


// TEST(ImageTest, 1DDynamicGaus3Clamp0) {

//     Image testHD("imgs/cat.jpeg");
//     Image target("imgs/tests/2Dgaus3cat0.jpeg");

//     ASSERT_NE(testHD.data, nullptr) << "Failed to load test image.";
//     ASSERT_NE(target.data, nullptr) << "Failed to load target image.";

//     Mask::GaussianDynamic1D gaussianBlur1(1, false);
//     Mask::GaussianDynamic1D gaussianBlur2(1, true);

//     OpenCLImageProcessor processor;

//     processor.std_convolve_clamp_to_0(testHD, &gaussianBlur1);
//     processor.std_convolve_clamp_to_0(testHD, &gaussianBlur2);
//     processor.diffmap(testHD, target);
//     // auto start = std::chrono::high_resolution_clock::now();

//     int is_black = is_image_black(testHD);

//     // auto end = std::chrono::high_resolution_clock::now();
//     // std::chrono::duration<double> elapsed = end - start;
//     // std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;


//     EXPECT_EQ(is_black, 1);
// }

// TEST(ImageTest, 1DDynamicGaus3Clampborder) {

//     Image testHD("imgs/cat.jpeg");
//     Image target("imgs/tests/2Dgaus3cat.jpeg");

//     ASSERT_NE(testHD.data, nullptr) << "Failed to load test image.";
//     ASSERT_NE(target.data, nullptr) << "Failed to load target image.";

//     Mask::GaussianDynamic1D gaussianBlur1(1, false);
//     Mask::GaussianDynamic1D gaussianBlur2(1, true);

//     OpenCLImageProcessor processor;

//     processor.std_convolve_clamp_to_border(testHD, &gaussianBlur1);
//     processor.std_convolve_clamp_to_border(testHD, &gaussianBlur2);
//     processor.diffmap(testHD, target);
//     // auto start = std::chrono::high_resolution_clock::now();

//     int is_black = is_image_black(testHD);

//     // auto end = std::chrono::high_resolution_clock::now();
//     // std::chrono::duration<double> elapsed = end - start;
//     // std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;


//     EXPECT_EQ(is_black, 1);
// }

TEST(ImageTest, local_binary_pattern) {

    Image testHD("imgs/cat.jpeg");

    ASSERT_NE(testHD.data, nullptr) << "Failed to load test image.";

    Image cpu = testHD;

    testHD.local_binary_pattern_cpu();

    OpenCLImageProcessor processor;

    processor.local_binary_pattern(cpu);
    processor.diffmap(testHD, cpu);
    // auto start = std::chrono::high_resolution_clock::now();

    int is_black = is_image_black(testHD);

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;


    EXPECT_EQ(is_black, 1);
}

TEST(ImageTest, simpleIntegralcpu) {

    // Create a simple 3x3 test image
    std::vector<int> pixels = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    std::vector<int> integratedPixels = {
        1, 3, 6,
        5, 12, 21,
        12, 27, 45
    };

    std::vector<int> rotated_pixels = {
        1, 2, 3,
        7, 11, 11,
        21, 29, 23
    };

    Image test(3, 3, 1);

    for (int i=0; i < 9; i++) {
        test.data[i] = pixels[i];
    }

    ASSERT_NE(test.data, nullptr) << "Failed to load test image.";

    std::unique_ptr<u_int32_t[]> integralImage(new uint32_t[3 * 3]);
    std::fill(integralImage.get(), integralImage.get() + 3 * 3, 0);
    std::unique_ptr<uint32_t[]> integralImageSobel = nullptr;
    std::unique_ptr<uint32_t[]> integralImageSquare = nullptr;
    std::unique_ptr<uint32_t[]> integralImageTilt(new uint32_t[3 * 3]);
    std::fill(integralImageTilt.get(), integralImageTilt.get() + 3 * 3, 0);

    test.integralImage_cpu(integralImage, integralImageSobel, integralImageSquare, integralImageTilt);


    EXPECT_EQ(integralImage[8], 45);

    for (int i = 0; i < 9; ++i) {
        // std::cout<<i * 3 + j<<i<<j<<"\n";
        EXPECT_EQ(integralImageTilt[i], rotated_pixels[i]);
        EXPECT_EQ(integralImage[i], integratedPixels[i]);
        
    }



    std::vector<int> pixels2 = {
        1,  4,  7,  4, 1,
        4, 16, 26, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1,  4,  7,  4, 1
    };

    Image test2(5, 5, 1);

    for (int i=0; i < 25; i++) {
        test2.data[i] = pixels2[i];
    }

    std::unique_ptr<u_int32_t[]> integralImage2(new uint32_t[5 * 5]);
    std::fill(integralImage2.get(), integralImage2.get() + 5 * 5, 0);
    integralImageSobel = nullptr;
    integralImageSquare = nullptr;
    integralImageTilt = nullptr;

    test2.integralImage_cpu(integralImage2, integralImageSobel, integralImageSquare, integralImageTilt);


    EXPECT_EQ(integralImage2[24], 273);
}

TEST(ImageTest, OpenCLIntegralsum) {

    // Create a simple 3x3 test image
    std::vector<int> pixels = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    std::vector<int> integratedPixels = {
        1, 3, 6,
        5, 12, 21,
        12, 27, 45
    };

    std::vector<int> rotated_pixels = {
        1, 2, 3,
        7, 11, 11,
        21, 29, 23
    };

    Image test(3, 3, 1);

    for (int i=0; i < 9; i++) {
        test.data[i] = pixels[i];
    }

    ASSERT_NE(test.data, nullptr) << "Failed to load test image.";

    std::unique_ptr<u_int32_t[]> integralImage(new uint32_t[3 * 3]);
    std::fill(integralImage.get(), integralImage.get() + 3 * 3, 0);
    std::unique_ptr<uint32_t[]> integralImageSobel = nullptr;
    std::unique_ptr<uint32_t[]> integralImageSquare(new uint32_t[3 * 3]);
    std::unique_ptr<uint32_t[]> integralImageSquare_cpu(new uint32_t[3 * 3]);
    std::unique_ptr<uint32_t[]> integralImageTilt = nullptr;
    // std::fill(integralImageTilt.get(), integralImageTilt.get() + 3 * 3, 0);

    OpenCLImageProcessor processor;
    test.integralImage_cpu(integralImage, integralImageSobel, integralImageSquare_cpu, integralImageTilt);
    processor.integralImage(test, integralImage, integralImageSquare, integralImageTilt, integralImageSobel);


    EXPECT_EQ(integralImage[8], 45);

    for (int i = 0; i < 9; ++i) {
        // std::cout<<i * 3 + j<<i<<j<<"\n";
        EXPECT_EQ(integralImageSquare[i], integralImageSquare_cpu[i]);
        EXPECT_EQ(integralImage[i], integratedPixels[i]);
        
    }

    integralImageSquare = nullptr;
    // Test if integralSumSquare is optional
    processor.integralImage(test, integralImage, integralImageSquare, integralImageTilt, integralImageSobel);
    EXPECT_EQ(integralImageSquare, nullptr);


    std::vector<int> pixels1 = {
        1, 2, 3, 5,
        4, 5, 6, 5,
        7, 8, 9, 5
    };

    Image test2(4, 3, 1);
    // Image test2("imgs/facetestgray.jpg");

    // for (int i=0; i < 12; i++) {
    //     test2.data[i] = pixels1[i];
    // }

    integralImage = std::make_unique<uint32_t[]>(test2.w * test2.h);
    integralImageSquare = std::make_unique<uint32_t[]>(test2.w * test2.h);
    integralImageSquare_cpu = std::make_unique<uint32_t[]>(test2.w * test2.h);

    test2.integralImage_cpu(integralImage, integralImageSobel, integralImageSquare_cpu, integralImageTilt);
    processor.integralImage(test2, integralImage, integralImageSquare, integralImageTilt, integralImageSobel);

    for (int i = 0; i < 12; ++i) {
        EXPECT_EQ(integralImageSquare[i], integralImageSquare_cpu[i]);
        
    }
}

#ifdef USE_CUDA
TEST(CudaTest, ResizeTest) {

    Image testHD("imgs/cat.jpeg");
    Image cpu = testHD;

    cpu.resizeBilinear_cpu(512, 512);

    CUDAImageProcessor cudap;
    cudap.resizeBilinear(testHD, 512, 512);

    cpu.diffmap_cpu(testHD);

    int is_black = is_image_black(cpu);

    EXPECT_EQ(is_black, 1);

}

TEST(CudaTest, integralImage) {

    Image testHD("imgs/tkl.jpg");

    std::unique_ptr<u_int32_t[]> integralImage(new uint32_t[testHD.w * testHD.h]);
    std::unique_ptr<uint32_t[]> integralImageSquare(new uint32_t[testHD.w * testHD.h]);
    std::unique_ptr<uint32_t[]> integralImageTilt(new uint32_t[testHD.w * testHD.h]);
    std::unique_ptr<uint32_t[]> integralImage_cpu(new uint32_t[testHD.w * testHD.h]);
    std::unique_ptr<uint32_t[]> integralImageSquare_cpu(new uint32_t[testHD.w * testHD.h]);
    std::unique_ptr<uint32_t[]> integralImageTilt_cpu(new uint32_t[testHD.w * testHD.h]);
    
    std::unique_ptr<uint32_t[]> integralImageSobel = nullptr;

    testHD.integralImage_cpu(integralImage_cpu, integralImageSobel, integralImageSquare_cpu, integralImageTilt_cpu);

    CUDAImageProcessor cudap;
    cudap.integralImage(testHD, integralImage, integralImageSquare, integralImageTilt);

    for (int i = 0; i < 12; ++i) {
        EXPECT_EQ(integralImageSquare[i], integralImageSquare_cpu[i]);
        EXPECT_EQ(integralImage[i], integralImage_cpu[i]);
        EXPECT_EQ(integralImageTilt[i], integralImageTilt_cpu[i]);
        
    }

}
#endif