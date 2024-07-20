#pragma once
#include "image.h"
#include "masks.h"
#include "opencl_image.h"
#include "haarCasscades.h"
#include <vector>
#include <unordered_map>

typedef struct Rect_ {

    int total;
    int width;
    int height;
    int x;
    int y;

} Rect;

class ViolaJones {
public:
    ViolaJones() {};
    ~ViolaJones() {};

    std::vector<Rect> detect(Image& image, std::vector<double> haar);
    std::vector<Rect> detect(Image& image, std::vector<double> haar, OpenCLImageProcessor& opencl);

    void draw(Image& image, std::vector<Rect> faces);
    bool intersect_rect(Rect rect1, Rect rect2);

    void set_edgeDensity(float n) {m_edgeDensity = n;}
    void set_scaleFactor(float n) {m_scaleFactor = n;}
    void set_stepSize(float n) {m_stepSize = n;}

private:
    float m_regions_overlap = 0.3;

    // Detection window variables
    float m_edgeDensity = 0.025;
    float m_initialScale = 1.0;
    float m_scaleFactor = 1.25;
    float m_stepSize = 1.5;

    std::vector<Rect> merge_rectangles(Image& image, std::vector<Rect> rects);
    bool evalStages(std::vector<double> haar, std::unique_ptr<u_int32_t[]>& integralImage, std::unique_ptr<u_int32_t[]>& integralImageSquare, std::unique_ptr<u_int32_t[]>& integralImageTilt, int i, int j, int width, int blockWidth, int blockHeight, float scale);

    bool edgeExclude(float edgeDensity, std::unique_ptr<u_int32_t[]>& integralImageSobel, int i, int j, int width, int blockWidth, int blockHeight);

};