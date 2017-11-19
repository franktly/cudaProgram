# cmake minimum required version
cmake_minimum_required(VERSION 2.8)

# cuda proj name
project(Optim2GaussianSmooth)

# packages
find_package(CUDA)

# set nvcc compile options
set(CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30; -G;-g)

# find cuda h file
# file(GLOB_RECURSE CURRENT_HEADERS *h *hpp *cuh)

# find cuda cpp and cu file
# file(GLOB_RECURSE CURRENT_SOURCES *cpp *cu)

# source_group("include" FILES ${CURRENT_HEADERS})
# source_group("source"  FILES ${CURRENT_SOURCES})

# exe cuda 
# cuda_add_executable(Optim2GaussianSmooth ${CURRENT_HEADERS} ${CURRENT_SOURCES})
cuda_add_executable(Optim2GaussianSmooth Optim2GaussianSmooth.cu)
	
