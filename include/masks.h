#pragma once
#include "pch.h"
#include <Eigen/Dense>



namespace Mask {

    class BaseMask {
    public:
        virtual ~BaseMask() = default;
        virtual int getWidth() const = 0;
        virtual int getHeight() const = 0;
        virtual int getCenterRow() const = 0;
        virtual int getCenterColumn() const = 0;
        virtual double getFilterFactor() const = 0;
        virtual const double* getData() const = 0;
    };

    class GaussianBlur3 : public BaseMask {
    private:
        static constexpr int width = 3;
        static constexpr int height = 3;
        static constexpr int cr = 1;
        static constexpr int cc = 1;
        static constexpr double filter_factor = 16;
        std::array<double, 9> mask;

    public:

        GaussianBlur3() {
            double filter[9] = {
            1.0, 2.0, 1.0,
            2.0, 4.0, 2.0,
            1.0, 2.0, 1.0
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask.data();
        }
    };

    class GaussianBlur5 : public BaseMask {
    private:
        static constexpr int width = 5;
        static constexpr int height = 5;
        static constexpr int cr = 2;
        static constexpr int cc = 2;
        static constexpr double filter_factor = 273;
        std::array<double, 25> mask;
    public:
        GaussianBlur5() {
            double filter[25] = {
            1.0f,  4.0f,  7.0f,  4.0f, 1.0f,
            4.0f, 16.0f, 26.0f, 16.0f, 4.0f,
            7.0f, 26.0f, 41.0f, 26.0f, 7.0f,
            4.0f, 16.0f, 26.0f, 16.0f, 4.0f,
            1.0f,  4.0f,  7.0f,  4.0f, 1.0f
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask.data();
        }
    };

    class SharpenMask : public BaseMask {
    private:
        static constexpr int width = 5;
        static constexpr int height = 5;
        static constexpr int cr = 2;
        static constexpr int cc = 2;
        static constexpr double filter_factor = 8.0;
        std::array<double, 25> mask;
    public:
        SharpenMask() {
            double filter[25] = {
            -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
            -1.0f,  2.0f,  2.0f,  2.0f, -1.0f,
            -1.0f,  2.0f,  8.0f,  2.0f, -1.0f,
            -1.0f,  2.0f,  2.0f,  2.0f, -1.0f,
            -1.0f, -1.0f, -1.0f, -1.0f, -1.0f
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask.data();
        }
    };

    class VertEdgeDetect : public BaseMask {
    private:
        static constexpr int width = 5;
        static constexpr int height = 5;
        static constexpr int cr = 2;
        static constexpr int cc = 2;
        static constexpr double filter_factor = 1.0;
        std::array<double, 25> mask;

    public:
        VertEdgeDetect() {
            double filter[25] = {
            0,  0, -1.0f,  0,  0,
            0,  0, -1.0f,  0,  0,
            0,  0,  4.0f,  0,  0,
            0,  0, -1.0f,  0,  0,
            0,  0, -1.0f,  0,  0
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask.data();
        }
    };

    class EdgeSharpen : public BaseMask {
    private:
        static constexpr int width = 3;
        static constexpr int height = 3;
        static constexpr int cr = 1;
        static constexpr int cc = 1;
        static constexpr double filter_factor = 1.0;
        std::array<double, 9> mask;

    public:

        EdgeSharpen() {
            double filter[9] = {
            1.0f,  1.0f, 1.0f,
            1.0f, -7.0f, 1.0f,
            1.0f,  1.0f, 1.0f
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask.data();
        }
    };

    class Emboss3D : public BaseMask {
    private:
        static constexpr int width = 3;
        static constexpr int height = 3;
        static constexpr int cr = 1;
        static constexpr int cc = 1;
        static constexpr double filter_factor = 1.0;
        std::array<double, 9> mask;

    public:

        Emboss3D() {
            double filter[9] = {
            2.0f,  0.0f,  0.0f,
            0.0f, -1.0f,  0.0f,
            0.0f,  0.0f, -1.0f
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask.data();
        }
    };

    class EdgeSobelX : public BaseMask {
    private:
        static constexpr int width = 3;
        static constexpr int height = 3;
        static constexpr int cr = 1;
        static constexpr int cc = 1;
        static constexpr double filter_factor = 1.0;
        std::array<double, 9> mask;

    public:

        EdgeSobelX() {
            double filter[9] = {
                -1.0f, 0.0f, 1.0f,
                -2.0f, 0.0f, 2.0f,
                -1.0f, 0.0f, 1.0f
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask.data();
        }
    };

    class EdgeSobelY : public BaseMask {
    private:
        static constexpr int width = 3;
        static constexpr int height = 3;
        static constexpr int cr = 1;
        static constexpr int cc = 1;
        static constexpr double filter_factor = 1.0;
        std::array<double, 9> mask;

    public:

        EdgeSobelY() {
            double filter[9] = {
                -1.0f, -2.0f, -1.0f,
                0.0f,  0.0f,  0.0f,
                1.0f,  2.0f,  1.0f
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask.data();
        }
    };

    class BoxBlur : public BaseMask {
    private:
        static constexpr int width = 3;
        static constexpr int height = 3;
        static constexpr int cr = 1;
        static constexpr int cc = 1;
        static constexpr double filter_factor = 9.0;
        std::array<double, 9> mask;

    public:

        BoxBlur() {
            double filter[9] = {
                1, 1, 1,
                1, 1, 1,
                1, 1, 1
            };

            for (int i = 0; i < width*height; ++i) {
                mask[i] = filter[i] / filter_factor;
            }
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask.data();
        }
    };

    class GaussianDynamic2D : public BaseMask {
    private:

        double sigma;
        int width;
        int height;
        int cr;
        int cc;
        double filter_factor = 1.0;
        std::vector<double> mask;

    public:

        GaussianDynamic2D(double sigma) {
            double kernel_radius = std::ceil(sigma) * 3;
            cr = cc = (int) kernel_radius;
            width = height = (int) kernel_radius * 2 + 1;

            // Generate matrix
            Eigen::VectorXf ax = Eigen::VectorXf::LinSpaced(width, -kernel_radius, kernel_radius);

            Eigen::MatrixXf xx(width, height);
            Eigen::MatrixXf yy(width, height);

            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    xx(i, j) = ax(j);
                    yy(i, j) = ax(i);
                }
            }

            Eigen::MatrixXf kernel = (-(xx.array().square() + yy.array().square()) / (2.0 * sigma * sigma)).exp();
            // Normalize the kernel
            kernel /= kernel.sum();

            // Flatten the 2D kernel matrix into a 1D vector
            mask.resize(width * height);
            for (int i = 0; i < width; ++i) {
                for (int j = 0; j < height; ++j) {
                    mask[i * width + j] = kernel(i, j);
                }
            }

        }

        GaussianDynamic2D() {
            GaussianDynamic2D(1);
        }

        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask.data();
        }
    };

    class GaussianDynamic1D : public BaseMask {
    private:

        double sigma;
        int width;
        int height;
        int cr;
        int cc;
        double filter_factor = 1.0;
        std::vector<double> mask;

    public:

        GaussianDynamic1D(double sigma, bool transpose) {
            double kernel_radius = std::ceil(sigma) * 3;
            int kernel_size = (int) kernel_radius * 2 + 1;
            if (transpose) {
                cr = (int) kernel_radius;
                cc = 0;
                height = kernel_size;
                width = 1;
            } else {
                cc = (int) kernel_radius;
                cr = 0;
                width = kernel_size;
                height = 1;
            }
            
            
            // Generate matrix
            Eigen::VectorXf ax = Eigen::VectorXf::LinSpaced(kernel_size, -kernel_radius, kernel_radius);

            Eigen::VectorXf kernel = (-(ax.array().square()) / (2.0f * sigma * sigma)).exp();
            // Normalize the kernel
            kernel /= kernel.sum();

            if (transpose) kernel = kernel.transpose();

            // Flatten the 2D kernel matrix into a 1D vector
            mask.resize(kernel_size);
            for (int i = 0; i < kernel_size; ++i) {
                mask[i] = static_cast<double>(kernel(i));
            }

        }


        int getWidth() const override {
            return width;
        }

        int getHeight() const override {
            return height;
        }

        int getCenterRow() const override {
            return cr;
        }

        int getCenterColumn() const override {
            return cc;
        }

        double getFilterFactor() const override {
            return filter_factor;
        }

        const double* getData() const override {
            return mask.data();
        }
    };

}