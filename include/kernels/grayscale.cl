__kernel void grayscale_lum(
    __global uchar* data, 
    __global uchar* output,
    int channels
) 
{
    int id = get_global_id(0);
    int pixelIndex = id * channels;

    uchar r = data[pixelIndex];
    uchar g = data[pixelIndex + 1];
    uchar b = data[pixelIndex + 2];
    uchar gray = (r * 0.2989 + g * 0.5870 + 0.1140 * b);

    output[id] = gray;
}

__kernel void grayscale_avg(
    __global uchar* data, 
    __global uchar* output,
    int channels
) 
{
    int id = get_global_id(0);
    int pixelIndex = id * channels;

    uchar r = data[pixelIndex];
    uchar g = data[pixelIndex + 1];
    uchar b = data[pixelIndex + 2];
    uchar gray = (r + g + b) / 3;

    output[id] = gray;
}
