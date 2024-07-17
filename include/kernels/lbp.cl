uchar computeLBP(int col, int row, int w, __global uchar* data) {
    int centerIndex = row * w + col;
    uchar center = data[centerIndex];
    uchar lbp = 0;
    lbp |= (data[(row-1) * w + (col-1)] >= center) << 7;
    lbp |= (data[(row-1) * w + (col)] >= center) << 6;
    lbp |= (data[(row-1) * w + (col+1)] >= center) << 5;
    lbp |= (data[(row) * w + (col+1)] >= center) << 4;
    lbp |= (data[(row+1) * w + (col+1)] >= center) << 3;
    lbp |= (data[(row+1) * w + (col)] >= center) << 2;
    lbp |= (data[(row+1) * w + (col-1)] >= center) << 1;
    lbp |= (data[(row) * w + (col-1)] >= center) << 0;
    return lbp;
}

__kernel void local_binary_output(
    __global uchar* data, 
    __global uchar* output,
    int w,
    int h
) 
{
    /* get global position in Y direction */
    int row = get_global_id(1);
    /* get global position in X direction */
    int col = get_global_id(0);


    if (row >= 1 && row < h-1 && col >= 1 && col < w-1) {
        uchar lbp = computeLBP(col, row, w, data);
        output[row * w + col] = lbp;
    }

}

