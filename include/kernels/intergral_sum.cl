__kernel void integralImage(
    __constant uchar *data,
    __global uint *integralImage,
    // __global uint *integralImageSquare,
    // __global uint *integralImageSobel,
    // __global uint *integralImageTilt,
    int width,
    int height
) 
{

    /* get global position in Y direction */
    int row = get_global_id(1);
    /* get global position in X direction */
    int col = get_global_id(0);

    if (row >= height || col >= width) {
        return;
    }

    // Debugging output
    // printf("Thread (%d, %d): Starting evaluation\n", row, col);
    
    

    // Row prefix sum
    for (int i = 0; i < width; i++) {
        if (i == 0) {
        integralImage[row * width + i] = data[row * width + i];
        // integralImageSquare[row * width + i] = data[row * width + i] * data[row * width + i];
        } else {
            integralImage[row * width + i] = data[row * width + i] + integralImage[row * width + i-1];
            // integralImageSquare[row * width + i] = (data[row * width + i] * data[row * width + i]) + integralImageSquare[row * width + i-1];
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Col prefix sum
    for (int j=0; j < height; j++) {
        integralImage[j * width + col] += integralImage[(j-1) * width + col];
    }

}