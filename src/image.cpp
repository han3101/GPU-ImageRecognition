#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define BYTE_BOUND(value) std::min(std::max((value), 0), 255)

#include "stb_image.h"
#include "stb_image_write.h"

#include "image.h"

Image::Image(const char* filename, int channel_force) {
	if(read(filename, channel_force)) {
		printf("Read %s\n", filename);
		size = w*h*channels;
	}
	else {
		printf("Failed to read %s\n", filename);
	}
}

Image::Image(int w, int h, int channels) : w(w), h(h), channels(channels) {
	size = w*h*channels;
	data = new uint8_t[size];
}

Image::Image(const Image& img) : Image(img.w, img.h, img.channels) {
	if (img.data) {
		memcpy(data, img.data, size);
	} else {
		data = nullptr;
	}
	
}

Image::~Image() {
	stbi_image_free(data);
}

bool Image::read(const char* filename, int channel_force) {
	data = stbi_load(filename, &w, &h, &channels, channel_force);
	channels = channel_force == 0 ? channels : channel_force;
	return data != NULL;
}

bool Image::write(const char* filename) {
	ImageType type = get_file_type(filename);
	int success;
  switch (type) {
    case PNG:
      success = stbi_write_png(filename, w, h, channels, data, w*channels);
      break;
    case BMP:
      success = stbi_write_bmp(filename, w, h, channels, data);
      break;
    case JPG:
      success = stbi_write_jpg(filename, w, h, channels, data, 100);
      break;
	case JPEG:
		success = stbi_write_jpg(filename, w, h, channels, data, 100);
      	break;

  }
  if(success != 0) {
    printf("\e[32mWrote \e[36m%s\e[0m, %d, %d, %d, %zu\n", filename, w, h, channels, size);
    return true;
  }
  else {
    printf("\e[31;1m Failed to write \e[36m%s\e[0m, %d, %d, %d, %zu\n", filename, w, h, channels, size);
    return false;
  }
}

ImageType Image::get_file_type(const char* filename) {
	const char* ext = strrchr(filename, '.');
	if(ext != nullptr) {
		if(strcmp(ext, ".png") == 0) {
			return PNG;
		}
		else if(strcmp(ext, ".jpg") == 0) {
			return JPG;
		}
		else if(strcmp(ext, ".bmp") == 0) {
			return BMP;
		}
		else if(strcmp(ext, ".jpeg") == 0) {
			return JPEG;
		}
		
	}
	return PNG;
}

Image& Image::grayscale_avg_cpu() {
	if(channels < 3) {
		printf("Image %p has less than 3 channels, it is assumed to already be grayscale. \n", this);
	}
	else {
		uint8_t* newData = new uint8_t[w*h];

		#pragma omp parallel for num_threads(4) schedule(static)
		for (int i = 0; i < w * h; ++i) {
            int gray = (data[i * channels] + data[i * channels + 1] + data[i * channels + 2]) / channels;
            newData[i] = static_cast<uint8_t>(gray);
        }
		// Delete the old data array
        delete[] data;

        // Update the image properties
        data = newData;
        size = w * h;
        channels = 1;
	}
	return *this;
}

Image& Image::grayscale_cpu() {

	if(channels < 3) {
		printf("Image %p has less than 3 channels, it is assumed to already be grayscale. \n", this);
	}
	else {
		uint8_t* newData = new uint8_t[w*h];
		
		#pragma omp parallel for num_threads(4) schedule(static)
		for (int i = 0; i < w * h; ++i) {
            int gray = 0.299 * data[i * channels] + 0.587 * data[i * channels + 1] + 0.114 * data[i * channels + 2];
            newData[i] = static_cast<uint8_t>(gray);
        }

		// Delete the old data array
        delete[] data;

        // Update the image properties
        data = newData;
        size = w * h;
        channels = 1;
	}
	return *this;
}


Image& Image::grayscale_lum_cpu() {

	if(channels < 3) {
		printf("Image %p has less than 3 channels, it is assumed to already be grayscale. \n", this);
	}
	else {
		uint8_t* newData = new uint8_t[w*h];
		
		#pragma omp parallel for num_threads(4) schedule(static)
		for (int i = 0; i < w * h; ++i) {
            int gray = 0.2126 * data[i * channels] + 0.7152 * data[i * channels + 1] + 0.0722 * data[i * channels + 2];
            newData[i] = static_cast<uint8_t>(gray);
        }

		// Delete the old data array
        delete[] data;

        // Update the image properties
        data = newData;
        size = w * h;
        channels = 1;
	}
	return *this;
}

Image& Image::diffmap_cpu(Image& img) {
	int compare_width = fmin(w,img.w);
	int compare_height = fmin(h,img.h);
	int compare_channels = fmin(channels,img.channels);
	for(uint32_t i=0; i<compare_height; ++i) {
		for(uint32_t j=0; j<compare_width; ++j) {
			for(uint8_t k=0; k<compare_channels; ++k) {
				data[(i*w+j)*channels+k] = BYTE_BOUND(abs(data[(i*w+j)*channels+k] - img.data[(i*img.w+j)*img.channels+k]));
			}
		}
	}
	return *this;
}

Image& Image::diffmap_scale_cpu(Image& img, uint8_t scl) {
	int compare_width = fmin(w,img.w);
	int compare_height = fmin(h,img.h);
	int compare_channels = fmin(channels,img.channels);
	uint8_t largest = 0;
	for(uint32_t i=0; i<compare_height; ++i) {
		for(uint32_t j=0; j<compare_width; ++j) {
			for(uint8_t k=0; k<compare_channels; ++k) {
				data[(i*w+j)*channels+k] = BYTE_BOUND(abs(data[(i*w+j)*channels+k] - img.data[(i*img.w+j)*img.channels+k]));
				largest = fmax(largest, data[(i*w+j)*channels+k]);
			}
		}
	}
	scl = 255/fmax(1, fmax(scl, largest));
	for(int i=0; i<size; ++i) {
		data[i] *= scl;
	}
	return *this;
}

Image& Image::flipX_cpu() {
	uint8_t tmp[4];
	uint8_t* px1;
	uint8_t* px2;
	for(int y = 0;y < h;++y) {
		for(int x = 0;x < w/2;++x) {
			px1 = &data[(x + y * w) * channels];
			px2 = &data[((w - 1 - x) + y * w) * channels];
			
			memcpy(tmp, px1, channels);
			memcpy(px1, px2, channels);
			memcpy(px2, tmp, channels);
		}
	}
	return *this;
}

Image& Image::flipY_cpu() {
	uint8_t tmp[4];
	uint8_t* px1;
	uint8_t* px2;
	for(int x = 0;x < w;++x) {
		for(int y = 0;y < h/2;++y) {
			px1 = &data[(x + y * w) * channels];
			px2 = &data[(x + (h - 1 - y) * w) * channels];

			memcpy(tmp, px1, channels);
			memcpy(px1, px2, channels);
			memcpy(px2, tmp, channels);
		}
	}
	return *this;
}


Image& Image::std_convolve_clamp_to_0_cpu(uint8_t channel, const Mask::BaseMask* mask) {
	
	std::vector<uint8_t> new_data(w*h);
	uint32_t ker_w = mask->getWidth(), ker_h = mask->getHeight(), cr = mask->getCenterRow(), cc = mask->getCenterColumn();
	const double* ker = mask->getData(); 

	uint64_t center = cr*ker_w + cc;

	#pragma omp parallel for num_threads(4) schedule(static)
	for(uint64_t k=channel; k<size; k+=channels) {
		double c = 0;
		for(long i = -((long)cr); i<(long)ker_h-cr; ++i) {
			long row = ((long)k/channels)/w-i;
			if(row < 0 || row > h-1) {
				continue;
			}
			for(long j = -((long)cc); j<(long)ker_w-cc; ++j) {
				long col = ((long)k/channels)%w-j;
				if(col < 0 || col > w-1) {
					continue;
				}
				c += ker[center+i*(long)ker_w+j]*data[(row*w+col)*channels+channel];
			}
		}

		new_data[k/channels] = (uint8_t)BYTE_BOUND((int)round(c));
	}

	for(uint64_t k=channel; k<size; k+=channels) {
		data[k] = new_data[k/channels];
	}
	return *this;
}

Image& Image::std_convolve_clamp_to_border_cpu(uint8_t channel, const Mask::BaseMask* mask) {
	std::vector<uint8_t> new_data(w*h);
	uint32_t ker_w = mask->getWidth(), ker_h = mask->getHeight(), cr = mask->getCenterRow(), cc = mask->getCenterColumn();
	const double* ker = mask->getData(); 

	uint64_t center = cr*ker_w + cc;

	#pragma omp parallel for num_threads(4) schedule(static)
	for(uint64_t k=channel; k<size; k+=channels) {
		double c = 0;
		for(long i = -((long)cr); i<(long)ker_h-cr; ++i) {
			long row = ((long)k/channels)/w-i;
			if(row < 0) {
				row = 0;
			}
			else if(row > h-1) {
				row = h-1;
			}
			for(long j = -((long)cc); j<(long)ker_w-cc; ++j) {
				long col = ((long)k/channels)%w-j;
				if(col < 0) {
					col = 0;
				}
				else if(col > w-1) {
					col = w-1;
				}
				c += ker[center+i*(long)ker_w+j]*data[(row*w+col)*channels+channel];
			}
		}
		new_data[k/channels] = (uint8_t)BYTE_BOUND((int)round(c));
	}

	for(uint64_t k=channel; k<size; k+=channels) {
		data[k] = new_data[k/channels];
	}
	return *this;
}

Image& Image::crop(uint16_t cx, uint16_t cy, uint16_t cw, uint16_t ch) {
	size = cw * ch * channels;
	uint8_t* croppedImage = new uint8_t[size];
	memset(croppedImage, 0, size);

	for(uint16_t y = 0;y < ch;++y) {
		if(y + cy >= h) {break;}
		for(uint16_t x = 0;x < cw;++x) {
			if(x + cx >= w) {break;}
			memcpy(&croppedImage[(x + y * cw) * channels], &data[(x + cx + (y + cy) * w) * channels], channels);
		}
	}

	w = cw;
	h = ch;
	

	delete[] data;
	data = croppedImage;
	croppedImage = nullptr;

	return *this;
}


Image& Image::resizeNN(uint16_t nw, uint16_t nh) {
	size = nw * nh * channels;
	uint8_t* newImage = new uint8_t[size];

	float scaleX = (float)nw / (w);
	float scaleY = (float)nh / (h);
	uint16_t sx, sy;

	for(uint16_t y = 0;y < nh;++y) {
		sy = (uint16_t)(y / scaleY);
		for(uint16_t x = 0;x < nw;++x) {
			sx = (uint16_t)(x / scaleX);

			memcpy(&newImage[(x + y * nw) * channels], &data[(sx + sy * w) * channels], channels);

		}
	}


	w = nw;
	h = nh;
	delete[] data;
	data = newImage;
	newImage = nullptr;

	return *this;
}

Image& Image::resizeBilinear_cpu(uint16_t nw, uint16_t nh) {
    size = nw * nh * channels;
    uint8_t* newImage = new uint8_t[size];

    float scaleX = (float)(w - 1) / (nw - 1);
    float scaleY = (float)(h - 1) / (nh - 1);
    int x, y, index;
    float fx, fy, fx1, fy1;
    uint16_t ix, iy, ix1, iy1;

    for (uint16_t ny = 0; ny < nh; ++ny) {
        fy = ny * scaleY;
        iy = (uint16_t)fy;
        fy1 = fy - iy;
        for (uint16_t nx = 0; nx < nw; ++nx) {
            fx = nx * scaleX;
            ix = (uint16_t)fx;
            fx1 = fx - ix;

            ix1 = ix + 1;
            iy1 = iy + 1;
            if (ix1 >= w) ix1 = ix;
            if (iy1 >= h) iy1 = iy;
			

            for (int c = 0; c < channels; ++c) {
                float value = (1 - fx1) * (1 - fy1) * data[(ix + iy * w) * channels + c] +
                              fx1 * (1 - fy1) * data[(ix1 + iy * w) * channels + c] +
                              (1 - fx1) * fy1 * data[(ix + iy1 * w) * channels + c] +
                              fx1 * fy1 * data[(ix1 + iy1 * w) * channels + c];

                newImage[(nx + ny * nw) * channels + c] = (uint8_t)value;
            }
        }
    }

    w = nw;
    h = nh;
    delete[] data;
    data = newImage;
    newImage = nullptr;

    return *this;
}

Image& Image::local_binary_pattern_cpu() {
	if(channels > 1) {
		std::cout<<"Image is not grayscale.<<\n";
		this->grayscale_avg_cpu();
	}

	uint8_t* newData = new uint8_t[w*h];

	// Exclude borders
	#pragma omp parallel for num_threads(4) schedule(static)
	for (int row=1; row < h-1; row++) {
		for (int col=1; col < w-1; col++) {
			uint8_t lbp = computeLBP(col, row);
			// std::cout<<"row: "<<row<<" col: "<<col<<" lbp: "<<(int)lbp<<"\n";
			newData[row * w + col] = lbp;
		}
	}

	delete[] data;
	data = newData;

	return *this;
}

void Image::integralImage_cpu(std::unique_ptr<u_int32_t[]>& integralImage, std::unique_ptr<u_int32_t[]>& integralImageSobel, std::unique_ptr<u_int32_t[]>& integralImageSquare, std::unique_ptr<u_int32_t[]>& integralImageTilt) {
	
	this->grayscale_cpu();
	Image sobel = *this;

	if (integralImageSobel) {
		Mask::EdgeSobelX sobelX;
    	Mask::EdgeSobelY sobelY;

		sobel.std_convolve_clamp_to_0_cpu(0, &sobelX);
		sobel.std_convolve_clamp_to_0_cpu(0, &sobelY);
	}
	
	
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			// Compute using summed area table (SAT) formula
			int index = i * w + j;

			if (integralImage) {
				integralImage[index] = data[index] 
                                + (i > 0 ? integralImage[index - w] : 0)
                                + (j > 0 ? integralImage[index - 1] : 0)
                                - (i > 0 && j > 0 ? integralImage[index - w - 1] : 0);
			}

			if (integralImageSquare) {
				integralImageSquare[index] = (data[index] * data[index]) 
                                + (i > 0 ? integralImageSquare[index - w] : 0)
                                + (j > 0 ? integralImageSquare[index - 1] : 0)
                                - (i > 0 && j > 0 ? integralImageSquare[index - w - 1] : 0);
			}

			if (integralImageSobel) {
				integralImageSobel[index] = sobel.data[index] 
                                + (i > 0 ? integralImageSobel[index - w] : 0)
                                + (j > 0 ? integralImageSobel[index - 1] : 0)
                                - (i > 0 && j > 0 ? integralImageSobel[index - w - 1] : 0);
			}

			if (integralImageTilt) {
				// Use rotated summed area table (RSAT) formula instead
				integralImageTilt[index] = data[index] 
								+ (i > 0 && j > 0? integralImageTilt[index - w - 1]: 0)
								+ (i > 0 && j < w-1? integralImageTilt[index - w + 1]: 0)
								- (i > 1? integralImageTilt[index - 2*w]: 0)
								+ (i > 0? data[index - w]: 0);
								
			}
			
		}
	}
}