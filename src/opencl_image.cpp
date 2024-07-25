#include "../include/opencl_image.h"
#include <fstream>
#include <iostream>
#include<cstdlib>


std::string OpenCLImageProcessor::getErrorString(cl_int error) {
    switch (error) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        default: return "Unknown OpenCL error";
    }
}

OpenCLImageProcessor::OpenCLImageProcessor() {
    init();
}

OpenCLImageProcessor::~OpenCLImageProcessor() {}

void OpenCLImageProcessor::init() {
    // Select Platform
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " No OpenCL platforms found.\n";
        exit(1);
    }

    cl::Platform default_platform = all_platforms[0];
#ifdef PROFILE
    std::cout << "Using platform: " <<default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
#endif

    // Select a device
    // TODO make this configurable
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found.\n";
        exit(1);
    }

    m_device = all_devices[0];
#ifdef PROFILE
    std::cout << "Using device: " << m_device.getInfo<CL_DEVICE_NAME>() << "\n";
#endif

    size_t max_work_group_size = m_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
#ifdef PROFILE
    std::cout << "Maximum work-group size: " << max_work_group_size << "\n";
#endif

    // Define properties for the command queue
#ifdef PROFILE
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
#else
    cl_command_queue_properties properties = 0;
#endif

    //create context, kernel source and queue to push commands to the device.
    m_context = cl::Context({ m_device });
    m_queue = cl::CommandQueue(m_context, m_device, properties);

}

std::string OpenCLImageProcessor::loadKernelSource(const std::string& fileName) {
    std::ifstream kernelFile(fileName);
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to load kernel." << std::endl;
        exit(1);
    }

    std::string sourceStr((std::istreambuf_iterator<char>(kernelFile)),
                           std::istreambuf_iterator<char>());
    kernelFile.close();

    return sourceStr;
}

void OpenCLImageProcessor::grayscale_avg(Image& image) {

    if(image.channels < 3) {
		std::cout<<"Image "<<&image<<" has less than 3 channels, it is assumed to already be grayscale."<<std::endl;
        return;
	}

    // Prepare memory
    size_t bytes_i = image.size * sizeof(uint8_t);
    size_t bytes_n = image.w * image.h;
    cl::Buffer data_d(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_i, image.data);
    cl::Buffer output_d(m_context, CL_MEM_WRITE_ONLY, bytes_n);

    // Load Kernel
    std::string kernel_code = loadKernelSource("include/kernels/grayscale.cl");
    //Appending the kernel, which is presented here as a string. 
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    // Compile program
    cl::Program program(m_context, sources);
    if (program.build({ m_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << "\n";
        exit(1);
    }

    // Load in kernel args
    cl::Kernel kernel(program, "grayscale_avg");
    kernel.setArg(0, data_d);
    kernel.setArg(1, output_d);
    kernel.setArg(2, image.channels);

    // Set dimensions
    cl::NDRange global(image.w * image.h);
    

#ifdef PROFILE
    // For Profiling
    cl::Event event;
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
#else
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
#endif

    m_queue.finish();

    uint8_t* newData = new uint8_t[bytes_n];
    delete[] image.data;
    image.data = newData;
    image.size = bytes_n;
    image.channels = 1;

    // Read back the results
    m_queue.enqueueReadBuffer(output_d, CL_TRUE, 0, bytes_n, image.data);

#ifdef PROFILE
    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << (double) elapsed_time / 1000000 << " ms" << std::endl;
#endif

}

void OpenCLImageProcessor::grayscale_lum(Image& image) {

    if(image.channels < 3) {
		std::cout<<"Image "<<&image<<" has less than 3 channels, it is assumed to already be grayscale."<<std::endl;
        return;
	}

    // Prepare memory
    size_t bytes_i = image.size * sizeof(uint8_t);
    size_t bytes_n = image.w * image.h;
    cl::Buffer data_d(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_i, image.data);
    cl::Buffer output_d(m_context, CL_MEM_WRITE_ONLY, bytes_n);

    // Load Kernel
    std::string kernel_code = loadKernelSource("include/kernels/grayscale.cl");
    //Appending the kernel, which is presented here as a string. 
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    // Compile program
    cl::Program program(m_context, sources);
    if (program.build({ m_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << "\n";
        exit(1);
    }

    // Load in kernel args
    cl::Kernel kernel(program, "grayscale_lum");
    kernel.setArg(0, data_d);
    kernel.setArg(1, output_d);
    kernel.setArg(2, image.channels);

    // Set dimensions
    cl::NDRange global(image.w * image.h);
    

#ifdef PROFILE
    // For Profiling
    cl::Event event;
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
#else
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
#endif

    m_queue.finish();

    uint8_t* newData = new uint8_t[bytes_n];
    delete[] image.data;
    image.data = newData;
    image.size = bytes_n;
    image.channels = 1;

    // Read back the results
    m_queue.enqueueReadBuffer(output_d, CL_TRUE, 0, bytes_n, image.data);

#ifdef PROFILE
    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << (double) elapsed_time / 1000000 << " ms" << std::endl;
#endif

}

void OpenCLImageProcessor::diffmap(Image& image1, Image& image2) {

    // Prepare memory
    size_t bytes_i = image1.size * sizeof(uint8_t);
    size_t bytes_o = image2.size * sizeof(uint8_t);
    cl::Buffer image1_d(m_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes_i, image1.data);
    cl::Buffer image2_d(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_o, image2.data);

    // Load Kernel
    std::string kernel_code = loadKernelSource("include/kernels/diffmap.cl");
    //Appending the kernel, which is presented here as a string. 
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    // Compile program
    cl::Program program(m_context, sources);
    if (program.build({ m_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << "\n";
        exit(1);
    }

    // Preprocessing
    int compare_width = fmin(image1.w,image2.w);
	int compare_height = fmin(image1.h,image2.h);
	int compare_channels = fmin(image1.channels,image2.channels);

    // Load in kernel args
    cl::Kernel kernel(program, "diffmap");
    kernel.setArg(0, image1_d);
    kernel.setArg(1, image2_d);
    kernel.setArg(2, image1.w);
    kernel.setArg(3, image1.h);
    kernel.setArg(4, image1.channels);
    kernel.setArg(5, image2.w);
    kernel.setArg(6, image2.h);
    kernel.setArg(7, image2.channels);
    kernel.setArg(8, compare_width);
    kernel.setArg(9, compare_height);
    kernel.setArg(10,compare_channels);

    // Set dimensions
    cl::NDRange global(image1.w, image1.h, image1.channels);

#ifdef PROFILE
    // For Profiling
    cl::Event event;
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
#else
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
#endif

    m_queue.finish();

    // Read back the results
    m_queue.enqueueReadBuffer(image1_d, CL_TRUE, 0, bytes_i, image1.data);

#ifdef PROFILE
    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << (double) elapsed_time / 1000000 << " ms" << std::endl;
#endif

}

void OpenCLImageProcessor::flipX(Image& image) {

    // Prepare memory
    size_t bytes_i = image.size * sizeof(uint8_t);
    cl::Buffer data_d(m_context, CL_MEM_READ_WRITE, bytes_i);
    m_queue.enqueueWriteBuffer(data_d, CL_TRUE, 0, bytes_i, image.data);

    // Load Kernel
    std::string kernel_code = loadKernelSource("include/kernels/flip.cl");
    //Appending the kernel, which is presented here as a string. 
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    // Compile program
    cl::Program program(m_context, sources);
    if (program.build({ m_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << "\n";
        exit(1);
    }

    // Load in kernel args
    cl::Kernel kernel(program, "flipX2d");
    kernel.setArg(0, data_d);
    kernel.setArg(1, image.w);
    kernel.setArg(2, image.h);
    kernel.setArg(3, image.channels);

    // Set dimensions
    cl::NDRange global(image.w, image.h);
    // cl::NDRange global(image.w, image.h, image.channels);
    

#ifdef PROFILE
    // For Profiling
    cl::Event event;
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
#else
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
#endif

    m_queue.finish();

    // Read back the results
    m_queue.enqueueReadBuffer(data_d, CL_TRUE, 0, bytes_i, image.data);

#ifdef PROFILE
    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << (double) elapsed_time / 1000000 << " ms" << std::endl;
#endif

}

void OpenCLImageProcessor::flipY(Image& image) {

    // Prepare memory
    size_t bytes_i = image.size * sizeof(uint8_t);
    cl::Buffer data_d(m_context, CL_MEM_READ_WRITE, bytes_i);
    m_queue.enqueueWriteBuffer(data_d, CL_TRUE, 0, bytes_i, image.data);

    // Load Kernel
    std::string kernel_code = loadKernelSource("include/kernels/flip.cl");
    //Appending the kernel, which is presented here as a string. 
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    // Compile program
    cl::Program program(m_context, sources);
    if (program.build({ m_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << "\n";
        exit(1);
    }

    // Load in kernel args
    cl::Kernel kernel(program, "flipY2d");
    kernel.setArg(0, data_d);
    kernel.setArg(1, image.w);
    kernel.setArg(2, image.h);
    kernel.setArg(3, image.channels);

    // Set dimensions
    cl::NDRange global(image.w, image.h);
    // cl::NDRange global(image.w, image.h, image.channels);
    

#ifdef PROFILE
    // For Profiling
    cl::Event event;
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
#else
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
#endif

    m_queue.finish();

    // Read back the results
    m_queue.enqueueReadBuffer(data_d, CL_TRUE, 0, bytes_i, image.data);

#ifdef PROFILE
    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << (double) elapsed_time / 1000000 << " ms" << std::endl;
#endif

}

void OpenCLImageProcessor::std_convolve_clamp_to_0(Image& image, const Mask::BaseMask* mask) {

    // Preprocessing for mask data
    // Mask offset is basically center row or center column
    uint32_t MASK_W = mask->getWidth(), MASK_OFFSET_W = mask->getCenterColumn();
    uint32_t MASK_H = mask->getHeight(), MASK_OFFSET_H = mask->getCenterRow();
	const double* ker = mask->getData(); 

    // Prepare memory
    size_t bytes_i = image.size * sizeof(uint8_t);
    size_t bytes_m = MASK_H * MASK_W * sizeof(double);
    cl::Buffer data_d(m_context, CL_MEM_READ_ONLY, bytes_i);
    cl::Buffer result_d(m_context, CL_MEM_WRITE_ONLY, bytes_i);
    cl::Buffer mask_d(m_context, CL_MEM_READ_ONLY, bytes_m);
    m_queue.enqueueWriteBuffer(data_d, CL_TRUE, 0, bytes_i, image.data);
    m_queue.enqueueWriteBuffer(mask_d, CL_TRUE, 0, bytes_m, ker);

    // Load Kernel
    std::string kernel_code = loadKernelSource("include/kernels/convolution.cl");
    //Appending the kernel, which is presented here as a string. 
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    // Compile program
    cl::Program program(m_context, sources);
    if (program.build({ m_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << "\n";
        exit(1);
    }

    // Load in kernel args
    cl::Kernel kernel(program, "convolution_0");
    kernel.setArg(0, data_d);
    kernel.setArg(1, result_d);
    kernel.setArg(2, mask_d);
    kernel.setArg(3, image.w);
    kernel.setArg(4, image.h);
    kernel.setArg(5, image.channels);
    kernel.setArg(6, MASK_W);
    kernel.setArg(7, MASK_H);
    kernel.setArg(8, MASK_OFFSET_W);
    kernel.setArg(9, MASK_OFFSET_H);

    // Set dimensions
    cl::NDRange global(image.w, image.h);
    // cl::NDRange global(image.w, image.h, image.channels);
    

#ifdef PROFILE
    // For Profiling
    cl::Event event;
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
#else
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
#endif

    m_queue.finish();

    // Read back the results
    m_queue.enqueueReadBuffer(result_d, CL_TRUE, 0, bytes_i, image.data);

#ifdef PROFILE
    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << (double) elapsed_time / 1000000 << " ms" << std::endl;
#endif

}

void OpenCLImageProcessor::std_convolve_clamp_to_border(Image& image, const Mask::BaseMask* mask) {

    // Preprocessing for mask data
    // Mask offset is basically center row or center column
    uint32_t MASK_W = mask->getWidth(), MASK_OFFSET_W = mask->getCenterColumn();
    uint32_t MASK_H = mask->getHeight(), MASK_OFFSET_H = mask->getCenterRow();
	const double* ker = mask->getData(); 

    // Prepare memory
    size_t bytes_i = image.size * sizeof(uint8_t);
    size_t bytes_m = MASK_H * MASK_W * sizeof(double);
    cl::Buffer data_d(m_context, CL_MEM_READ_ONLY, bytes_i);
    cl::Buffer result_d(m_context, CL_MEM_WRITE_ONLY, bytes_i);
    cl::Buffer mask_d(m_context, CL_MEM_READ_ONLY, bytes_m);
    m_queue.enqueueWriteBuffer(data_d, CL_TRUE, 0, bytes_i, image.data);
    m_queue.enqueueWriteBuffer(mask_d, CL_TRUE, 0, bytes_m, ker);

    // Load Kernel
    std::string kernel_code = loadKernelSource("include/kernels/convolution.cl");
    //Appending the kernel, which is presented here as a string. 
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    // Compile program
    cl::Program program(m_context, sources);
    if (program.build({ m_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << "\n";
        exit(1);
    }

    // Load in kernel args
    cl::Kernel kernel(program, "convolution_border");
    kernel.setArg(0, data_d);
    kernel.setArg(1, result_d);
    kernel.setArg(2, mask_d);
    kernel.setArg(3, image.w);
    kernel.setArg(4, image.h);
    kernel.setArg(5, image.channels);
    kernel.setArg(6, MASK_W);
    kernel.setArg(7, MASK_H);
    kernel.setArg(8, MASK_OFFSET_W);
    kernel.setArg(9, MASK_OFFSET_H);

    // Set dimensions
    cl::NDRange global(image.w, image.h);
    // cl::NDRange global(image.w, image.h, image.channels);
    

#ifdef PROFILE
    // For Profiling
    cl::Event event;
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
#else
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
#endif

    m_queue.finish();

    // Read back the results
    m_queue.enqueueReadBuffer(result_d, CL_TRUE, 0, bytes_i, image.data);

#ifdef PROFILE
    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << (double) elapsed_time / 1000000 << " ms" << std::endl;
#endif

}

// BROKEN
void OpenCLImageProcessor::std_convolve_clamp_to_cyclic(Image& image, const Mask::BaseMask* mask) {
    if(image.channels < 3) {
		std::cout<<"Image "<<&image<<" has less than 3 channels which is the required channel for this convolution, please use other methods."<<std::endl;
        return;
	}

    // Preprocessing for mask data
    // Mask offset is basically center row or center column
    uint32_t MASK_DIM = mask->getWidth(), MASK_OFFSET = mask->getCenterRow();
	const double* ker = mask->getData(); 

    // We will be using OpenCL's image format
    cl_int ret;
    cl::ImageFormat imageFormat(CL_RGBA, CL_UNSIGNED_INT8);
    cl::Image2D inputImage_d(m_context, CL_MEM_READ_ONLY, imageFormat, (size_t)image.w, (size_t)image.h, 0, nullptr, &ret);
    cl::Image2D output_d(m_context, CL_MEM_WRITE_ONLY, imageFormat, (size_t)image.w, (size_t)image.h, 0, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        std::cerr << "clCreateImage2D error: " << ret << "\n";
        return;
    }

    std::array<size_t, 3> origin = {0, 0, 0};
    std::array<size_t, 3> region = {(size_t)image.w, (size_t)image.h, 1};


    ret = m_queue.enqueueWriteImage(inputImage_d, CL_TRUE, origin, region, 0, 0, image.data);
    if (ret != CL_SUCCESS) {
        std::cerr << "WriteImage error: " << getErrorString(ret) << "\n";
        return;
    }


    // Use sampler to handle masking conditions
    // Create sampler using C standard due to bug
    // CL_ADDRESS_REPEAT is a circular clamp
    // Define sampler properties
    cl_sampler_properties sampler_properties[] = {
        CL_SAMPLER_NORMALIZED_COORDS, CL_FALSE,
        CL_SAMPLER_ADDRESSING_MODE, CL_ADDRESS_REPEAT,
        CL_SAMPLER_FILTER_MODE, CL_FILTER_LINEAR,
        0
    };

    // Create a sampler with specified properties using the C API
    cl_sampler samplerC = clCreateSamplerWithProperties(
        m_context(),
        sampler_properties,
        &ret
    );
    if (ret != CL_SUCCESS) {
        std::cerr << "clSampler error: " << ret << "\n";
        return;
    }

    cl::Sampler sampler(samplerC);

    // Debugging: Print sampler properties
    // cl_addressing_mode addressingMode;
    // sampler.getInfo(CL_SAMPLER_ADDRESSING_MODE, &addressingMode);
    // std::cout << "Sampler Addressing Mode: " << addressingMode << std::endl;

    // cl_filter_mode filterMode;
    // sampler.getInfo(CL_SAMPLER_FILTER_MODE, &filterMode);
    // std::cout << "Sampler Filter Mode: " << filterMode << std::endl;

    // cl_bool normalizedCoords;
    // sampler.getInfo(CL_SAMPLER_NORMALIZED_COORDS, &normalizedCoords);
    // std::cout << "Sampler Normalized Coordinates: " << normalizedCoords << std::endl;

    // Prepare memory
    size_t bytes_i = image.size * sizeof(uint8_t);
    size_t bytes_m = MASK_DIM * MASK_DIM * sizeof(double);
    cl::Buffer mask_d(m_context, CL_MEM_READ_ONLY, bytes_m);
    m_queue.enqueueWriteBuffer(mask_d, CL_TRUE, 0, bytes_m, ker);

    // Load Kernel
    std::string kernel_code = loadKernelSource("include/kernels/convolution.cl");
    //Appending the kernel, which is presented here as a string. 
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    // Compile program
    cl::Program program(m_context, sources);
    if (program.build({ m_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << "\n";
        exit(1);
    }

    // Load in kernel args
    cl::Kernel kernel(program, "convolution_circular");
    kernel.setArg(0, inputImage_d);
    kernel.setArg(1, output_d);
    kernel.setArg(2, mask_d);
    kernel.setArg(3, sampler);
    kernel.setArg(4, MASK_DIM);
    kernel.setArg(5, MASK_OFFSET);

    // Set dimensions
    cl::NDRange global(image.w, image.h);
    // cl::NDRange local(8, 8);
    

#ifdef PROFILE
    // For Profiling
    cl::Event event;
    ret = m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to enqueue kernel: " << ret << "\n";
        return;
    }
#else
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
#endif

    m_queue.finish();

    // Read back the results
    m_queue.enqueueReadImage(output_d, CL_TRUE, origin, region, 0, 0, image.data);

#ifdef PROFILE
    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << (double) elapsed_time / 1000000 << " ms" << std::endl;
#endif

}

void OpenCLImageProcessor::resizeBilinear(Image& image, int nw, int nh) {

    // Prepare memory
    cl_int ret;
    size_t bytes_i = image.size * sizeof(uint8_t);
    size_t bytes_o = nw * nh * image.channels * sizeof(uint8_t);
    cl::Buffer data_d(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_i, image.data);
    cl::Buffer output_d(m_context, CL_MEM_WRITE_ONLY, bytes_o);

    // Load Kernel
    std::string kernel_code = loadKernelSource("include/kernels/resize.cl"); 
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    // Compile program
    cl::Program program(m_context, sources);
    if (program.build({ m_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << "\n";
        exit(1);
    }

    float scaleX = (float) (image.w-1) / (nw-1);
    float scaleY = (float) (image.h-1) / (nh-1);

    // Load in kernel args
    cl::Kernel kernel(program, "resize_bilinear");
    kernel.setArg(0, data_d);
    kernel.setArg(1, output_d);
    kernel.setArg(2, nw);
    kernel.setArg(3, nh);
    kernel.setArg(4, image.w);
    kernel.setArg(5, image.h);
    kernel.setArg(6, image.channels);
    kernel.setArg(7, scaleX);
    kernel.setArg(8, scaleY);

    // Set dimensions
    cl::NDRange global(nw, nh);
    

#ifdef PROFILE
    // For Profiling
    cl::Event event;
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
#else
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
#endif

    m_queue.finish();

    // Read back the results
    image.size = nw * nh * image.channels;
	uint8_t* newImage = new uint8_t[image.size];
    image.w = nw;
	image.h = nh;
	delete[] image.data;
	image.data = newImage;
	newImage = nullptr;
    ret = m_queue.enqueueReadBuffer(output_d, CL_TRUE, 0, bytes_o, image.data);
    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to read out buffer: " << ret << "\n";
        return;
    }
#ifdef PROFILE
    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << (double) elapsed_time / 1000000 << " ms" << std::endl;
#endif

}

void OpenCLImageProcessor::resizeBicubic(Image& image, int nw, int nh) {

    // Prepare memory
    cl_int ret;
    size_t bytes_i = image.size * sizeof(uint8_t);
    size_t bytes_o = nw * nh * image.channels * sizeof(uint8_t);
    cl::Buffer data_d(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_i, image.data);
    cl::Buffer output_d(m_context, CL_MEM_WRITE_ONLY, bytes_o);

    // Load Kernel
    std::string kernel_code = loadKernelSource("include/kernels/resize.cl"); 
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    // Compile program
    cl::Program program(m_context, sources);
    if (program.build({ m_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << "\n";
        exit(1);
    }

    float scaleX = (float) (image.w-1) / (nw-1);
    float scaleY = (float) (image.h-1) / (nh-1);

    // Load in kernel args
    cl::Kernel kernel(program, "resize_bicubic");
    kernel.setArg(0, data_d);
    kernel.setArg(1, output_d);
    kernel.setArg(2, nw);
    kernel.setArg(3, nh);
    kernel.setArg(4, image.w);
    kernel.setArg(5, image.h);
    kernel.setArg(6, image.channels);
    kernel.setArg(7, scaleX);
    kernel.setArg(8, scaleY);

    // Set dimensions
    cl::NDRange global(nw, nh);
    

#ifdef PROFILE
    // For Profiling
    cl::Event event;
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
#else
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
#endif

    m_queue.finish();

    // Read back the results
    image.size = nw * nh * image.channels;
	uint8_t* newImage = new uint8_t[image.size];
    image.w = nw;
	image.h = nh;
	delete[] image.data;
	image.data = newImage;
	newImage = nullptr;
    ret = m_queue.enqueueReadBuffer(output_d, CL_TRUE, 0, bytes_o, image.data);
    if (ret != CL_SUCCESS) {
        std::cerr << "Failed to read out buffer: " << ret << "\n";
        return;
    }
#ifdef PROFILE
    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << (double) elapsed_time / 1000000 << " ms" << std::endl;
#endif

}

void OpenCLImageProcessor::local_binary_pattern(Image& image) {

    if(image.channels > 1) {
		std::cout<<"Image is not grayscale."<<"\n";
        image.grayscale_avg_cpu();
	}

    // Prepare memory
    size_t bytes_i = image.size * sizeof(uint8_t);
    cl::Buffer data_d(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_i, image.data);
    cl::Buffer output_d(m_context, CL_MEM_WRITE_ONLY, bytes_i);

    // Load Kernel
    std::string kernel_code = loadKernelSource("include/kernels/lbp.cl");
    //Appending the kernel, which is presented here as a string. 
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    // Compile program
    cl::Program program(m_context, sources);
    if (program.build({ m_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << "\n";
        exit(1);
    }

    // Load in kernel args
    cl::Kernel kernel(program, "local_binary_output");
    kernel.setArg(0, data_d);
    kernel.setArg(1, output_d);
    kernel.setArg(2, image.w);
    kernel.setArg(3, image.h);

    // Set dimensions
    cl::NDRange global(image.w, image.h);
    

#ifdef PROFILE
    // For Profiling
    cl::Event event;
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
#else
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
#endif

    m_queue.finish();

    // Read back the results
    m_queue.enqueueReadBuffer(output_d, CL_TRUE, 0, bytes_i, image.data);

#ifdef PROFILE
    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << (double) elapsed_time / 1000000 << " ms" << std::endl;
#endif

}

void OpenCLImageProcessor::evalStages(Image& image, std::vector<double>& haar, std::vector<int>& results, std::unique_ptr<u_int32_t[]>& integralImage, std::unique_ptr<u_int32_t[]>& integralImageSquare, std::unique_ptr<u_int32_t[]>& integralImageTilt, int blockWidth, int blockHeight, float scale, float inverseArea, int step) {

    // Prepare memory
    size_t bytes_h = haar.size() * sizeof(double);
    size_t bytes_i = (image.w * image.h) * sizeof(uint32_t);
    size_t bytes_r = ((image.h) * (image.w) * sizeof(int));
    cl::Buffer haar_d(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_h, haar.data());
    cl::Buffer integralImage_d(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_i, integralImage.get());
    cl::Buffer integralImageSquare_d(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_i, integralImageSquare.get());
    cl::Buffer integralImageTilt_d(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_i, integralImageTilt.get());
    cl::Buffer result_d(m_context, CL_MEM_WRITE_ONLY, bytes_r);

    // Load Kernel
    std::string kernel_code = loadKernelSource("include/kernels/viola_jones.cl");
    //Appending the kernel, which is presented here as a string. 
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    // Compile program
    cl::Program program(m_context, sources);
    if (program.build({ m_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << "\n";
        exit(1);
    }

    // Load in kernel args
    cl::Kernel kernel(program, "evalStages");
    kernel.setArg(0, haar_d);
    kernel.setArg(1, result_d);
    kernel.setArg(2, integralImage_d);
    kernel.setArg(3, integralImageSquare_d);
    kernel.setArg(4, integralImageTilt_d);
    kernel.setArg(5, (int) haar.size());
    kernel.setArg(6, image.w);
    kernel.setArg(7, image.h);
    kernel.setArg(8, blockWidth);
    kernel.setArg(9, blockHeight);
    kernel.setArg(10, scale);
    kernel.setArg(11, inverseArea);
    kernel.setArg(12, step);

    // Set dimensions
    cl::NDRange global(image.w, image.h);
    

#ifdef PROFILE
    // For Profiling
    cl::Event event;
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
#else
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
#endif

    m_queue.finish();

    // Read back the results
    m_queue.enqueueReadBuffer(result_d, CL_TRUE, 0, bytes_r, results.data());

#ifdef PROFILE
    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << (double) elapsed_time / 1000000 << " ms" << std::endl;
#endif

}

void OpenCLImageProcessor::integralImage(Image& image, std::unique_ptr<u_int32_t[]>& integralImage, std::unique_ptr<u_int32_t[]>& integralImageSquare, std::unique_ptr<u_int32_t[]>& integralImageTilt,  std::unique_ptr<u_int32_t[]>& integralImageSobel) {
    
    if (image.channels > 1) {
        this->grayscale_lum(image);
    }

    // Prepare memory
    size_t bytes_i = (image.w * image.h) * sizeof(uint32_t);
    size_t bytes_d = (image.h * image.w * sizeof(uint8_t));
    cl::Buffer data_d(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_d, image.data);
    cl::Buffer integralImage_d(m_context, CL_MEM_WRITE_ONLY, bytes_i);
    cl::Buffer debugBuffer_d(m_context, CL_MEM_WRITE_ONLY, 3 * sizeof(uint32_t));

    // Load Kernel
    std::string kernel_code = loadKernelSource("include/kernels/integral_sum.cl");
    //Appending the kernel, which is presented here as a string. 
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(),kernel_code.length() });

    // Compile program
    cl::Program program(m_context, sources);
    if (program.build({ m_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device) << "\n";
        exit(1);
    }

    // Load in kernel args
    cl::Kernel kernel(program, "integralImage");
    kernel.setArg(0, data_d);
    kernel.setArg(1, integralImage_d);
    kernel.setArg(2, image.w);
    kernel.setArg(3, image.h);
    kernel.setArg(4, debugBuffer_d);


    // if (integralImageSobel) {
    //     Image sobel = image;
    //     Mask::EdgeSobelX sobelX;
    // 	Mask::EdgeSobelY sobelY;
    //     this->std_convolve_clamp_to_0(sobel, &sobelX);
    //     this->std_convolve_clamp_to_0(sobel, &sobelY);
    //     integralImageSobel_d = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, bytes_i);
    //     cl::Buffer sobel_d(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes_d, sobel.data);
    //     kernelSobel = cl::Kernel(program, "integralImage");
    //     kernelSobel.setArg(0, sobel_d);
    //     kernelSobel.setArg(1, integralImageSobel_d);
    //     kernelSobel.setArg(2, image.w);
    //     kernelSobel.setArg(3, image.h);
    // }

    // Set dimensions
    cl::NDRange global(image.w, image.h);
    

#ifdef PROFILE
    // For Profiling
    cl::Event event;
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
#else
    m_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
#endif

    m_queue.finish();

    // Read back the results
    std::vector<uint32_t> debugBuffer(3, 1);
    m_queue.enqueueReadBuffer(integralImage_d, CL_TRUE, 0, bytes_i, integralImage.get());
    m_queue.enqueueReadBuffer(debugBuffer_d, CL_TRUE, 0, 3 * sizeof(uint32_t), debugBuffer.data());

    std::cout << "Debug info: " << debugBuffer[0] << ", " << debugBuffer[1] << ", " << debugBuffer[2] << "\n";

#ifdef PROFILE
    // Get profiling information
    cl_ulong time_start;
    cl_ulong time_end;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);

    // Compute the elapsed time in nanoseconds
    cl_ulong elapsed_time = time_end - time_start;

    std::cout << "Kernel execution time: " << (double) elapsed_time / 1000000 << " ms" << std::endl;
#endif

}