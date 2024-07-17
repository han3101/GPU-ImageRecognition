__kernel void grayscale_avg(
    __global uchar* data, 
    int channels
) 
{
    int id = get_global_id(0);
    int pixelIndex = id * channels;

    uchar r = data[pixelIndex];
    uchar g = data[pixelIndex + 1];
    uchar b = data[pixelIndex + 2];
    uchar gray = (r + g + b) / 3;

    data[pixelIndex] = gray;
    data[pixelIndex + 1] = gray;
    data[pixelIndex + 2] = gray;
}
