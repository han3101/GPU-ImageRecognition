#include "viola_jones.h"
#include "disjointset.h"


std::vector<Rect> ViolaJones::detect(Image& image, std::vector<_Float64> haar) {

    int total = 0;
    std::vector<Rect> rects;
    std::unique_ptr<u_int32_t[]> integralImage(new uint32_t[image.w * image.h]);
    std::unique_ptr<u_int32_t[]> integralImageSquare(new uint32_t[image.w * image.h]);
    std::unique_ptr<uint32_t[]> integralImageSobel = nullptr;
    std::unique_ptr<uint32_t[]> integralImageTilt(new uint32_t[image.w * image.h]);

    if (m_edgeDensity > 0) {
        integralImageSobel = std::make_unique<uint32_t[]>(image.w * image.h);
    }

    image.integralImage_cpu(integralImage, integralImageSobel, integralImageSquare, integralImageTilt);
    
    _Float64 minWidth = haar[0];
    _Float64 minHeight = haar[1];
    float scale = m_initialScale * m_scaleFactor;
    int blockWidth = static_cast<int>(scale * minWidth);
    int blockHeight = static_cast<int>(scale * minHeight);

    while (blockWidth < image.w && blockHeight < image.h) {
        int step = static_cast<int>(scale * m_stepSize);
        
        for (int i=0; i<(image.h-blockHeight); i++) {
            for (int j=0; j<(image.w-blockWidth); j++) {

                // if (m_edgeDensity > 0) {
                //     if (this->edgeExclude(m_edgeDensity, integralImageSobel, i, j, image.w, blockWidth, blockHeight)) {
                //         continue;
                //     }
                // }
                
                if (this->evalStages(haar, integralImage, integralImageSquare, integralImageTilt, i, j, image.w, blockWidth, blockHeight, scale)) {
                    total++;
                    rects.emplace_back(Rect{
                        0,
                        blockWidth,
                        blockHeight,
                        j,
                        i
                    });
                }
            }

   
        }
        scale *= m_scaleFactor;
        blockWidth = static_cast<int>(scale * minWidth);
        blockHeight = static_cast<int>(scale * minHeight);
    }
    
    return this->merge_rectangles(image, rects);
}


std::vector<Rect> ViolaJones::merge_rectangles(Image& image, std::vector<Rect> rects) {

    int num_recs = rects.size();
    std::cout<<"num rectangles "<<num_recs<<"\n";
    DisjointSet disjointset(num_recs);

    for (int i=0; i<num_recs; i++) {
        Rect r1 = rects[i];
        for (int j=0; j<num_recs; j++) {
            Rect r2 = rects[j];

            if (this->intersect_rect(r1, r2)) {
                int x1 = std::max(r1.x, r2.x);
                int y1 = std::max(r1.y, r2.y);
                int x2 = std::max(r1.x + r1.width, r2.x + r2.width);
                int y2 = std::max(r1.y + r1.height, r2.y + r2.height);
                float overlap = (x1 - x2) * (y1 - y2);
                int area1 = (r1.width * r1.height);
                int area2 = (r2.width * r2.height);

                if ((overlap / (area1 * (area1 / area2)) >= this->m_regions_overlap)
                && (overlap / (area2 * (area1 / area2)) >=  this->m_regions_overlap)) {
                    disjointset.unite(i, j);
                }
            }
        }
    }

    std::unordered_map<int, Rect> map;
    for (int k=0; k < disjointset.size(); k++) {
        int rec = disjointset.find(k);

        if (0 == map.count(rec)) {
            map.emplace(rec, Rect{1, rects[k].width, rects[k].height, rects[k].x, rects[k].y});
        } else {
            map[rec].total++;
            map[rec].width += rects[k].width;
            map[rec].height += rects[k].height;
            map[rec].x += rects[k].x;
            map[rec].y += rects[k].y;
        }
    }

    std::vector<Rect> result;
    for (const auto& entry: map) {
        const Rect& rect = entry.second;
        Rect averagedRect = {
            rect.total,
            static_cast<int>(rect.width / rect.total + 0.5),  
            static_cast<int>(rect.height / rect.total + 0.5),
            static_cast<int>(rect.x / rect.total + 0.5),
            static_cast<int>(rect.y / rect.total + 0.5)
        };
        result.push_back(averagedRect);
    }

    return result;
}

bool ViolaJones::intersect_rect(Rect rect1, Rect rect2) {

    int x0 = rect1.x, y0 = rect1.y;
    int x1 = rect1.x + rect1.width, y1 = rect1.y + rect1.height;
    int x2 = rect2.x, y2 = rect2.y;
    int x3 = rect2.x + rect2.width, y3 = rect2.y + rect2.height;

    return !(x2 > x1 || x3 < x0 || y2 > y1 || y3 < y0);
}


bool ViolaJones::evalStages(std::vector<_Float64> haar, std::unique_ptr<u_int32_t[]>& integralImage, std::unique_ptr<u_int32_t[]>& integralImageSquare, std::unique_ptr<u_int32_t[]>& integralImageTilt, int i, int j, int width, int blockWidth, int blockHeight, int scale) {

    float inverseArea = 1.0 / (blockWidth * blockHeight);
    int wba = i * width + j;
    int wbb = wba + blockWidth;
    int wbd = wba + blockHeight * width;
    int wbc = wbd + blockWidth;

    _Float32 mean = (integralImage[wba] - integralImage[wbb] - integralImage[wbd] + integralImage[wbc]) * inverseArea;
    _Float32 variance = (integralImageSquare[wba] - integralImageSquare[wbb] - integralImageSquare[wbd] + integralImageSquare[wbc]) * inverseArea - mean * mean;

    _Float32 standardDeviation = 1;
    if (variance > 0) {
        standardDeviation = std::sqrt(variance);
    }

    int length = haar.size();

    for (int w = 2; w < length;) {
        float stageSum = 0;
        auto stageThreshold = haar[w++];
        auto nodeLength = haar[w++];

        while (nodeLength--) {
            float rectsSum = 0.0;
            auto tilted = haar[w++];
            auto rectsLength = haar[w++];

            for (int r = 0; r < rectsLength; r++) {
                int rectLeft = static_cast<int>(j + haar[w++] * scale + 0.5);
                int rectTop = static_cast<int>(i + haar[w++] * scale + 0.5); 
                int rectWidth = static_cast<int>(haar[w++] * scale + 0.5);
                int rectHeight = static_cast<int>(haar[w++] * scale + 0.5);

                auto rectWeight = haar[w++];
                int w1, w2, w3, w4;

                if (tilted) {
                    // Use Rotated sum area table
                    w1 = (rectLeft - rectHeight + rectWidth) + (rectTop + rectWidth + rectHeight - 1) * width;
                    w2 = rectLeft + (rectTop - 1) * width;
                    w3 = (rectLeft - rectHeight) + (rectTop + rectHeight - 1) * width;
                    w4 = (rectLeft + rectWidth) + (rectTop + rectWidth - 1) * width;
                    rectsSum += (integralImageTilt[w1] + integralImageTilt[w2] - integralImageTilt[w3] - integralImageTilt[w4]) * rectWeight;
                    // std::cout<<"tilt"<<"\n";
                } else {
                    // Sum area table
                    w1 = rectTop * width + rectLeft;
                    w2 = w1 + rectWidth;
                    w3 = w1 + rectHeight * width;
                    w4 = w3 + rectWidth;
                    rectsSum += (integralImage[w1] - integralImage[w2] - integralImage[w3] + integralImage[w4]) * rectWeight;
                }

            }

            auto nodeThreshold = haar[w++];
            auto nodeLeft = haar[w++];
            auto nodeRight = haar[w++];

            if (rectsSum * inverseArea < nodeThreshold * standardDeviation) {
                stageSum += nodeLeft;
            } else {
                stageSum += nodeRight;
            }
        }

        if (stageSum < stageThreshold) {
            return false;
        }
    }

    return true;
}


void ViolaJones::draw(Image& image, std::vector<Rect> faces) {

    for (const Rect& face: faces) {
        // Draw top and bottom borders
        for (int j = face.x; j < face.x + face.width; j++) {
            int topIndex = (face.y * image.w + j) * image.channels;
            int bottomIndex = ((face.y + face.height - 1) * image.w + j) * image.channels;

            if (topIndex + image.channels - 1 < image.size) {
                image.data[topIndex] = 255;
                if (image.channels > 1) image.data[topIndex + 1] = 0;
                if (image.channels > 2) image.data[topIndex + 2] = 0;
                if (image.channels > 3) image.data[topIndex + 3] = 255;
            }

            if (bottomIndex + image.channels - 1 < image.size) {
                image.data[bottomIndex] = 255;
                if (image.channels > 1) image.data[bottomIndex + 1] = 0;
                if (image.channels > 2) image.data[bottomIndex + 2] = 0;
                if (image.channels > 3) image.data[bottomIndex + 3] = 255;
            }
        }

        // Draw left and right borders
        for (int i = face.y; i < face.y + face.height; i++) {
            int leftIndex = (i * image.w + face.x) * image.channels;
            int rightIndex = (i * image.w + face.x + face.width - 1) * image.channels;

            if (leftIndex + image.channels - 1 < image.size) {
                image.data[leftIndex] = 255;
                if (image.channels > 1) image.data[leftIndex + 1] = 0;
                if (image.channels > 2) image.data[leftIndex + 2] = 0;
                if (image.channels > 3) image.data[leftIndex + 3] = 255;
            }

            if (rightIndex + image.channels - 1 < image.size) {
                image.data[rightIndex] = 255;
                if (image.channels > 1) image.data[rightIndex + 1] = 0;
                if (image.channels > 2) image.data[rightIndex + 2] = 0;
                if (image.channels > 3) image.data[rightIndex + 3] = 255;
            }
        }
    }
}

bool ViolaJones::edgeExclude(float edgeDensity, std::unique_ptr<u_int32_t[]>& integralImageSobel, int i, int j, int width, int blockWidth, int blockHeight) {
    int wba = i * width + j;
    int wbb = wba + blockWidth;
    int wbd = wba + blockHeight * width;
    int wbc = wbd + blockWidth;
    
    float blockEdgeDensity = (float) (integralImageSobel[wba] - integralImageSobel[wbb] - integralImageSobel[wbd] + integralImageSobel[wbc]) / (blockWidth * blockHeight * 255);

    if (blockEdgeDensity < edgeDensity) return true;
    // std::cout<<blockEdgeDensity<<"\n";

    return false;
}