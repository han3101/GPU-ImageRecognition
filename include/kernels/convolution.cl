__kernel void convolution_0(
    __global uchar *matrix,
    __global uchar *result,
    __constant double* mask,
    int w,
    int h,
    int channels,
    int mask_w,
    int mask_h,
    int mask_offset_w,
    int mask_offset_h
) 
{
    /* get global position in Y direction */
    int row = get_global_id(1);
    /* get global position in X direction */
    int col = get_global_id(0);

    for (int ch = 0; ch < channels; ++ch) {
        // Starting index for calculation
        int start_r = row - mask_offset_h;
        int start_c = col - mask_offset_w;

        // Temp value for accumulating result
        double temp = 0;

        for (int i = 0; i < mask_h; i++) {
            for (int j = 0; j < mask_w; ++j) {
                if ((start_r + i) >= 0 && (start_r + i) < h) {
                    if ((start_c + j) >= 0 && (start_c + j) < w) {
                        // Accumulate results
                        temp += matrix[((start_r + i) * w + (start_c + j)) * channels + ch] * mask[i * mask_w + j];
                    }
                }
            }
        }
        // Write back the result
        result[(row * w + col) * channels + ch] = (uchar)clamp((int)round(temp), 0, 255);
    }
}



__kernel void convolution_border(
    __global uchar *matrix,
    __global uchar *result,
    __constant double* mask,
    int w,
    int h,
    int channels,
    int mask_w,
    int mask_h,
    int mask_offset_w,
    int mask_offset_h
) 
{
    /* get global position in Y direction */
    int row = get_global_id(1);
    /* get global position in X direction */
    int col = get_global_id(0);

    for (int ch = 0; ch < channels; ++ch) {
        // Starting index for calculation
        int start_r = row - mask_offset_h;
        int start_c = col - mask_offset_w;

        // Temp value for accumulating result
        double temp = 0;

        // Iterate over all the rows
        for (int i = 0; i < mask_h; i++) {
            int r = start_r + i;
            // Range check for rows
            if ((start_r + i) < 0) {
                r = 0;
            } else if ((start_r + i) >= h) {
                r = h - 1;
            }

            // Go over column
            for (int j = 0; j < mask_w; ++j) {
                int c = start_c + j;
                // Range check for cols
                if ((start_c + j) < 0) {
                    c = 0;
                } else if ((start_c + j) >= w) {
                    c = w - 1;
                }
                // Accumulate results
                temp += matrix[(r * w + c) * channels + ch] * mask[i * mask_w + j];
            }
        }
        // Write back the result
        result[(row * w + col) * channels + ch] = (uchar)clamp((int)round(temp), 0, 255);
    }
}

// ONLY DOES 3 CHANNELS
__kernel void convolution_circular(
    __read_only image2d_t inputImage,
    __write_only image2d_t result,
    __constant double* mask,
    sampler_t sampler,
    int MASK_DIM,
    int MASK_OFFSET
) 
{
    /* get global position in Y direction */
    int row = get_global_id (1);
    /* get global position in X direction */
    int col = get_global_id (0);

    // Starting index for calculation
    int start_r = row - MASK_OFFSET;
    int start_c = col - MASK_OFFSET;

    // Initialize sum for each channel
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    // Coordinates for accessing the image
    int2 coords; 

    // Iterate rows
    for (int i=0; i < MASK_DIM; ++i) {
        coords.y = start_r + i;

        // Iterate columns
        for (int j=0; j < MASK_DIM; j++) {
            coords.x = start_c + j;

            // Read a pixel from the image
            float4 pixel = read_imagef(inputImage, sampler, coords);

            // Apply mask to each of the 3 channels
            int maskId = i * MASK_DIM + j;
            sum.x += pixel.x * mask[maskId];
            sum.y += pixel.y * mask[maskId];
            sum.z += pixel.z * mask[maskId];
        }
    }

    // Copy the data to the output image
    // set coords back to original
    coords.x = col;
    coords.y = row;

    float4 clampedSum;
    clampedSum.x = (uchar)clamp((int)round(sum.x), 0, 255);
    clampedSum.y = (uchar)clamp((int)round(sum.y), 0, 255);
    clampedSum.z = (uchar)clamp((int)round(sum.z), 0, 255);

    // clampedSum.x = (uchar) 0;
    // clampedSum.y = (uchar) 0;
    // clampedSum.z = (uchar) 0;

    write_imagef(result, coords, clampedSum);

}