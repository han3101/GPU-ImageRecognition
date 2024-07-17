#include <gtest/gtest.h>

#include "image.h"
#include "opencl_image.h"
#include "masks.h"
#include <cstdlib>
#include <iostream>

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

TEST(ImageTest, 2DDynamicGaus3) {

    Image testHD("imgs/cat.jpeg");
    Image target("imgs/tests/2Dgaus3cat.jpeg");

    ASSERT_NE(testHD.data, nullptr) << "Failed to load test image.";
    ASSERT_NE(target.data, nullptr) << "Failed to load target image.";

    Mask::GaussianDynamic2D gaussianBlur((int) 1);

    OpenCLImageProcessor processor;
    processor.std_convolve_clamp_to_border(testHD, &gaussianBlur);

    processor.diffmap(testHD, target);

    auto start = std::chrono::high_resolution_clock::now();

    int is_black = is_image_black(testHD);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;


    EXPECT_EQ(is_black, 1);
}

TEST(ImageTest, 1DDynamicGaus3_cpu) {

    Image testHD("imgs/cat.jpeg");
    Image target("imgs/tests/2Dgaus3cat.jpeg");

    ASSERT_NE(testHD.data, nullptr) << "Failed to load test image.";
    ASSERT_NE(target.data, nullptr) << "Failed to load target image.";

    Mask::GaussianDynamic1D gaussianBlur1(1, false);
    Mask::GaussianDynamic1D gaussianBlur2(1, true);

    auto start = std::chrono::high_resolution_clock::now();

    testHD.std_convolve_clamp_to_border_cpu(0, &gaussianBlur1);
    testHD.std_convolve_clamp_to_border_cpu(1, &gaussianBlur1);
    testHD.std_convolve_clamp_to_border_cpu(2, &gaussianBlur1);

    testHD.std_convolve_clamp_to_border_cpu(0, &gaussianBlur2);
    testHD.std_convolve_clamp_to_border_cpu(1, &gaussianBlur2);
    testHD.std_convolve_clamp_to_border_cpu(2, &gaussianBlur2);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;

    OpenCLImageProcessor processor;

    processor.diffmap(testHD, target);
    // auto start = std::chrono::high_resolution_clock::now();

    int is_black = is_image_black(testHD);

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;


    EXPECT_EQ(is_black, 1);
}

TEST(ImageTest, 1DDynamicGaus3Clamp0) {

    Image testHD("imgs/cat.jpeg");
    Image target("imgs/tests/2Dgaus3cat0.jpeg");

    ASSERT_NE(testHD.data, nullptr) << "Failed to load test image.";
    ASSERT_NE(target.data, nullptr) << "Failed to load target image.";

    Mask::GaussianDynamic1D gaussianBlur1(1, false);
    Mask::GaussianDynamic1D gaussianBlur2(1, true);

    OpenCLImageProcessor processor;

    processor.std_convolve_clamp_to_0(testHD, &gaussianBlur1);
    processor.std_convolve_clamp_to_0(testHD, &gaussianBlur2);
    processor.diffmap(testHD, target);
    // auto start = std::chrono::high_resolution_clock::now();

    int is_black = is_image_black(testHD);

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;


    EXPECT_EQ(is_black, 1);
}

TEST(ImageTest, 1DDynamicGaus3Clampborder) {

    Image testHD("imgs/cat.jpeg");
    Image target("imgs/tests/2Dgaus3cat.jpeg");

    ASSERT_NE(testHD.data, nullptr) << "Failed to load test image.";
    ASSERT_NE(target.data, nullptr) << "Failed to load target image.";

    Mask::GaussianDynamic1D gaussianBlur1(1, false);
    Mask::GaussianDynamic1D gaussianBlur2(1, true);

    OpenCLImageProcessor processor;

    processor.std_convolve_clamp_to_border(testHD, &gaussianBlur1);
    processor.std_convolve_clamp_to_border(testHD, &gaussianBlur2);
    processor.diffmap(testHD, target);
    // auto start = std::chrono::high_resolution_clock::now();

    int is_black = is_image_black(testHD);

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Time taken for computation: " << elapsed.count() * 1000 << " ms" << std::endl;


    EXPECT_EQ(is_black, 1);
}