// #include "cuda_image.cuh"
#include "cuda_image.cuh"

/*
* For optimization use with cudaMemcpyToSymbol
* Constant image size might need to be bigger
*/
// __constant__ unsigned char image_data[1024 * 1024];

template <typename T>
__device__ T clamp(T x, T a, T b) {
    return max(a, min(x, b));
}

__global__ void grayscale_avg_cu(const unsigned char *data, unsigned char *output, int channels) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    int pixelIndex = id * channels;

    unsigned char r = data[pixelIndex];
    unsigned char g = data[pixelIndex + 1];
    unsigned char b = data[pixelIndex + 2];

    unsigned char gray = (r + g + b) / 3;

    output[id] = gray;

} 

__global__ void grayscale_lum_cu(const unsigned char *data, unsigned char *output, int channels) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    int pixelIndex = id * channels;

    unsigned char r = data[pixelIndex];
    unsigned char g = data[pixelIndex + 1];
    unsigned char b = data[pixelIndex + 2];

    unsigned char gray = (r * 0.299 + g * 0.5870 + 0.1140 * b);

    output[id] = gray;
} 


__global__ void flipX_cu(unsigned char *data, int w, int h, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w / 2 && y < h) {
        int left = (x + y*w) * channels;
        int right = ((w - 1- x) + y*w) * channels;

        for (int c=0; c < channels; ++c) {
            unsigned char tmp = data[left + c];
            data[left + c] = data[right + c];
            data[right + c] = tmp;
        }
    }
}

__global__ void flipY_cu(unsigned char *data, int w, int h, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h / 2) {
        int left = (x + y*w) * channels;
        int right = (x + (h - y + 1)*w) * channels;

        for (int c=0; c < channels; ++c) {
            unsigned char tmp = data[left + c];
            data[left + c] = data[right + c];
            data[right + c] = tmp;
        }
    }
}


__global__ void flipYvector_cu(uchar3 *data, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h / 2) {
        int left = x + y * w;
        int right = (x + (h - y + 1) * w);

        uchar3 tmp = data[left];
        data[left] = data[right];
        data[right] = tmp;
    }
}

__global__ void resize_bilinear_cu(unsigned char *data, unsigned char *output, int nw, int nh, int w, int h, int channels, float scaleX, float scaleY) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < nh && col < nw) {
        float float_pos_x, float_pos_y, offset_x, offset_y;
        ushort low_x, low_y, high_x, high_y; 

        float_pos_x = col * scaleX;
        float_pos_y = row * scaleY;

        low_x = (ushort)floor(float_pos_x);
        low_y = (ushort)floor(float_pos_y);
        high_x = low_x + 1;
        high_y = low_y + 1;

        if (high_x >= w) {
            high_x = low_x;
        }
        if (high_y >= h) {
            high_y = low_y;
        }

        offset_x = float_pos_x - low_x;
        offset_y = float_pos_y - low_y;

        for (int c = 0; c < channels; ++c) {
            
            float value = (1-offset_x) * (1-offset_y) * data[(low_x + low_y * w) * channels + c] +
                        offset_x * (1-offset_y) * data[(high_x + low_y * w) * channels + c] +
                        (1-offset_x) * offset_y * data[(low_x + high_y * w) * channels + c] +
                        offset_x * offset_y * data[(high_x + high_y * w) * channels + c];

            output[(col + row * nw) * channels + c] = clamp(value, 0.0f, 255.0f);
        }


    }
}