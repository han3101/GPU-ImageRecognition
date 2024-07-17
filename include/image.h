#pragma once
#include <stdint.h>
#include <cstdio>
#include <complex>
#include <iostream>
#include <vector>
#include <omp.h>
#include "masks.h"


//legacy feature of C
#undef __STRICT_ANSI__
#define _USE_MATH_DEFINES 
#include <cmath>
#ifndef M_PI
	#define M_PI (3.14159265358979323846)
#endif

enum ImageType {
	PNG, JPG, BMP, JPEG
};


struct Image {
	uint8_t* data = NULL;
	size_t size = 0;
	int w;
	int h;
	int channels;

	Image(const char* filename, int channel_force = 0);
	Image(int w, int h, int channels = 3);
	Image(const Image& img);
	~Image();

	bool write(const char* filename);

private:
	bool read(const char* filename, int channel_force = 0);

	void mask_calc(double* mask, double filter_factor, int w, int h) {
		for (int i = 0; i < w*h; ++i) {
			mask[i] = mask[i] / filter_factor;
		}
	}
	

public:
	ImageType get_file_type(const char* filename);

	Image& grayscale_avg_cpu();
	Image& grayscale_lum_cpu();

	Image& diffmap_cpu(Image& img);
	Image& diffmap_scale_cpu(Image& img, uint8_t scl = 0);

	Image& flipX_cpu();
	Image& flipY_cpu();

	Image& std_convolve_clamp_to_0_cpu(uint8_t channel, const Mask::BaseMask* mask);
	Image& std_convolve_clamp_to_border_cpu(uint8_t channel, const Mask::BaseMask* mask);
	
	Image& crop(uint16_t cx, uint16_t cy, uint16_t cw, uint16_t ch);
	Image& resizeNN(uint16_t nw, uint16_t nh);
	Image& resizeBilinear_cpu(uint16_t nw, uint16_t nh);

};
