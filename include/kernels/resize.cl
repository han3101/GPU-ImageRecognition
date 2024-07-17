__kernel void resize_bilinear(
    __global uchar* data, 
    __global uchar* output,
    int nw,
    int nh,
    int w, 
    int h,
    int channels,
    float scaleX,
    float scaleY
) 
{
    int row = get_global_id(1);
    int col = get_global_id(0);

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

            output[(col + row * nw) * channels + c] = (uchar) clamp(value, 0.0f, 255.0f);
        }
    }
    
}


// --------- Bicubic Interpolation -----------


float cubicInterpolate(float p[4], float t) {
    return p[1] + 0.5f * t * (p[2] - p[0] + t * (2.0f * p[0] - 5.0f * p[1] + 4.0f * p[2] - p[3] + t * (3.0f * (p[1] - p[2]) + p[3] - p[0])));
}

float bicubicInterpolate(float p[4][4], float x, float y) {
    float arr[4];

    arr[0] = cubicInterpolate(p[0], y);
    arr[1] = cubicInterpolate(p[1], y);
    arr[2] = cubicInterpolate(p[2], y);
    arr[3] = cubicInterpolate(p[3], y);
    return cubicInterpolate(arr, x);
}


__kernel void resize_bicubic(
    __global uchar* data, 
    __global uchar* output,
    int nw,
    int nh,
    int w, 
    int h,
    int channels,
    float scaleX,
    float scaleY
) 
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row < nh && col < nw) {

        float float_pos_x, float_pos_y, delta_x, delta_y;
        ushort low_x, low_y;

        float_pos_x = col * scaleX;
        float_pos_y = row * scaleY;

        low_x = (ushort)floor(float_pos_x);
        low_y = (ushort)floor(float_pos_y);

        delta_x = float_pos_x - low_x;
        delta_y = float_pos_y - low_y;

        for (int c = 0; c < channels; ++c) {
            float filter[4][4];
            
            for (int i=-1; i<3; i++) {
                for (int j=-1; j<3; j++) {
                    int x = clamp(low_x+j, 0, w-1);
                    int y = clamp(low_y+i, 0, h-1);
                    filter[i+1][j+1] = data[(low_x + low_y * w) * channels + c];
                }
            }

            float interpolatedValue = bicubicInterpolate(filter, delta_x, delta_y);

            output[(col + row * nw) * channels + c] = (uchar) clamp(interpolatedValue, 0.0f, 255.0f);
        }
    }
    
}