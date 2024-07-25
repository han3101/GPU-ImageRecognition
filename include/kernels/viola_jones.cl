__kernel void evalStages(
    __global double *haar,
    __global int *result,
    __constant uint *integralImage,
    __constant uint *integralImageSquare,
    __constant uint *integralImageTilt,
    int haar_size,
    int width,
    int height,
    int block_w,
    int block_h,
    float scale,
    float inverseArea,
    int step,
    float edgeDensity,
    __constant uint *integralImageSobel
) 
{
    /* get global position in Y direction */
    int row = get_global_id(1) * step;
    /* get global position in X direction */
    int col = get_global_id(0) * step;

    if (row >= (height - block_h) || col >= (width - block_w)) {
        return;
    }

    // Debugging output
    // printf("Thread (%d, %d): Starting evaluation\n", row, col);
    
    int wba = row * width + col;
    int wbb = wba + block_w;
    int wbd = wba + block_h * width;
    int wbc = wbd + block_w;

    // Edge Exclusion
    if (edgeDensity > 0) {
        float blockEdgeDensity = (float) (integralImageSobel[wba] - integralImageSobel[wbb] - integralImageSobel[wbd] + integralImageSobel[wbc]) / (block_w * block_h * 255);
        if (blockEdgeDensity < edgeDensity) return;
    }
    

    float mean = (integralImage[wba] - integralImage[wbb] - integralImage[wbd] + integralImage[wbc]) * inverseArea;
    float variance = (integralImageSquare[wba] - integralImageSquare[wbb] - integralImageSquare[wbd] + integralImageSquare[wbc]) * inverseArea - mean * mean;

    float standardDeviation = 1;
    if (variance > 0) {
        standardDeviation = sqrt(variance);
    }

    int pass = 1;

    for (int w = 2; w < haar_size; ) {
        double stageSum = 0.0f;
        double stageThreshold = haar[w++];
        double nodeLength = haar[w++];

        while (nodeLength--) {
            float rectsSum = 0.0f;
            double tilted = haar[w++];
            double rectsLength = haar[w++];

            for (int r = 0; r < rectsLength; r++) {
                int rectLeft = (int) (col + haar[w++] * scale + 0.5);
                int rectTop = (int) (row + haar[w++] * scale + 0.5); 
                int rectWidth = (int) (haar[w++] * scale + 0.5);
                int rectHeight = (int) (haar[w++] * scale + 0.5);

                double rectWeight = haar[w++];
                int w1, w2, w3, w4;

                if (tilted) {
                    // Use Rotated sum area table
                    w1 = (rectLeft - rectHeight + rectWidth) + (rectTop + rectWidth + rectHeight - 1) * width;
                    w2 = rectLeft + (rectTop - 1) * width;
                    w3 = (rectLeft - rectHeight) + (rectTop + rectHeight - 1) * width;
                    w4 = (rectLeft + rectWidth) + (rectTop + rectWidth - 1) * width;
                    rectsSum += (integralImageTilt[w1] + integralImageTilt[w2] - integralImageTilt[w3] - integralImageTilt[w4]) * rectWeight;
                } else {
                    // Sum area table
                    w1 = rectTop * width + rectLeft;
                    w2 = w1 + rectWidth;
                    w3 = w1 + rectHeight * width;
                    w4 = w3 + rectWidth;
                    rectsSum += (integralImage[w1] - integralImage[w2] - integralImage[w3] + integralImage[w4]) * rectWeight;
                }

            }

            double nodeThreshold = haar[w++];
            double nodeLeft = haar[w++];
            double nodeRight = haar[w++];

            if (rectsSum * inverseArea < nodeThreshold * standardDeviation) {
                stageSum += nodeLeft;
            } else {
                stageSum += nodeRight;
            }
        }

        if (stageSum < stageThreshold) {
            pass = 0;
            break;
        }
    }

    result[row * width + col] = pass ? 1: 0;
}