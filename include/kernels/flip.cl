__kernel void flipX(
    __global uchar* data, 
    int w,
    int h,
    int channels
) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    if (x < w / 2 && y < h && z < channels) {
        int left = ((x + y*w) * channels) + z;
        int right = (((w - 1- x) + y*w) * channels) + z;

        uchar tmp = data[left];
        data[left] = data[right];
        data[right] = tmp;
    }
    
}

__kernel void flipX2d(
    __global uchar* data, 
    int w,
    int h,
    int channels
) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < w / 2 && y < h) {
        int left = (x + y*w) * channels;
        int right = ((w - 1- x) + y*w) * channels;

        for (int c=0; c < channels; ++c) {
            uchar tmp = data[left + c];
            data[left + c] = data[right + c];
            data[right + c] = tmp;
        }
    }
    
}

__kernel void flipY2d(
    __global uchar* data, 
    int w,
    int h,
    int channels
) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < w  && y < h / 2) {
        int left = (x + y*w) * channels;
        int right = (x + (h - y + 1)*w) * channels;

        for (int c=0; c < channels; ++c) {
            uchar tmp = data[left + c];
            data[left + c] = data[right + c];
            data[right + c] = tmp;
        }
    }
    
}


