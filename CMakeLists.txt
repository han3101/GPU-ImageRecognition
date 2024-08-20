cmake_minimum_required(VERSION 3.30.0)
project(ImageRecognition VERSION 0.1.1 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)



# For profiling performance
if(PROFILE)
    add_definitions(-DPROFILE)
endif()

add_definitions(-DCL_TARGET_OPENCL_VERSION=220)

enable_testing()

# Check for CUDA
# find_package(CUDA QUIET)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    # Set the CUDA standard
    # set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.3/bin/nvcc)
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80")
    
    enable_language(CUDA)
    message(STATUS "CUDA Toolkit found, setting custom paths for OpenCL.")
    # set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} /usr/local/cuda-12.3/targets/x86_64-linux/include)
    # set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/local/cuda-12.3/targets/x86_64-linux/lib64)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    link_directories(${CUDAToolkit_LIBRARY_DIRS})


    # Automatically detect the GPU architecture
    set(NVCC_FLAGS "-O2")

    # Preprocessor macro for CUDA
    add_definitions(-DUSE_CUDA)
else()
    message(STATUS "CUDA Toolkit not found, using default OpenCL paths.")
endif()

# OPENCL requirements
find_package(OpenCL QUIET)
if (OpenCL_FOUND)
    message(STATUS "OpenCL found")
    add_definitions(-DUSE_OPENCL)
else()
    message(WARNING "OpenCL not found. OpenCL-related functionality will be disabled.")
endif()

# Build program
set(APPLICATION_SOURCE 
    src/main.cpp
    src/image.cpp
    src/viola_jones.cpp
)

if(OpenCL_FOUND)
    list(APPEND APPLICATION_SOURCE
        src/opencl_image.cpp
    )
endif()

if(CUDAToolkit_FOUND)
    list(APPEND APPLICATION_SOURCE
        src/cuda_image.cu
        src/cuda_kernels.cu
    )
endif()

set(APPLICATION_HEADERS 
    include/image.h
    include/stb_image_write.h
    include/stb_image.h
    include/opencl_image.h
    include/masks.h
    include/viola_jones.h
    include/haarCasscades.h
)

add_executable(${PROJECT_NAME} ${APPLICATION_SOURCE})

set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/output")

set(INCLUDE_DIRS /user/include ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRS})

target_link_directories(${PROJECT_NAME} PRIVATE /usr/lib/x86_64-linux-gnu)

# OPEN MP
find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
if (OpenCL_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCL_LIBRARIES})
endif()


set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Create output directories if they do not exist
add_custom_target(create_output_dir ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
)

add_dependencies(${PROJECT_NAME} create_output_dir)

# Custom commands for clean
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}
)

# Custom target for 'r'
add_custom_target(r
    COMMAND ${CMAKE_COMMAND} -E echo "Running ${PROJECT_NAME}"
    COMMAND ${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)


# For testing
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/e39786088138f2749d64e9e90e0f9902daa77c40.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)
set(TEST_SOURCE 
    src/test.cc
    src/image.cpp
    src/viola_jones.cpp
)

if(OpenCL_FOUND)
    list(APPEND TEST_SOURCE
        src/opencl_image.cpp
    )
endif()

if(CUDAToolkit_FOUND)
    list(APPEND TEST_SOURCE
        src/cuda_image.cu
        src/cuda_kernels.cu
    )
endif()

add_executable(${PROJECT_NAME}_test ${TEST_SOURCE})
target_include_directories(${PROJECT_NAME}_test PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}_test PRIVATE ${OpenCL_LIBRARIES} OpenMP::OpenMP_CXX GTest::gtest_main)

gtest_discover_tests(${PROJECT_NAME}_test
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTIES ENVIRONMENT "GTEST_COLOR=1"
)
