#pragma once

#include <cuda_runtime.h>
#include "stdio.h"
#include <sys/time.h>

// for almost all non-kernel functions , they will return cudaError_t 
// just use HANDLE_CUDA_ERROR(function) to check cuda error
static void HandleError(cudaError_t err, const char *file, int line)
{
	if(cudaSuccess !=err)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err),file, line);
		exit(EXIT_FAILURE);
	}

}

#define HANDLE_CUDA_ERROR(err) (HandleError(err, __FILE__, __LINE__))


// for the kernel function, just use cudaGetLastError() behind the kernel function like this to check cuda error:
/*
 * cudaError_t err = cudaGetLastError()
 * printf("%s\n", cudaGetErrorString(err));
 */

#define TIME_TRACE_CPU_START(traceName) \
struct timeval start_time_##traceName; \
struct timeval stop_time_##traceName; \
gettimeofday(&start_time_##traceName, 0); 

#define TIME_TRACE_CPU_STOP(traceName) \
gettimeofday(&stop_time_##traceName, 0); \
float fElapsedTimeViaCPU_##traceName =  1000.0 * (stop_time_##traceName.tv_sec - start_time_##traceName.tv_sec) + (0.001 * (stop_time_##traceName.tv_usec - start_time_##traceName.tv_usec)); \
printf("%s elapsed time via CPU is %f ms.\n", #traceName, fElapsedTimeViaCPU_##traceName); 

#define TIME_TRACE_CUDA_EVENT_START(traceName) \
cudaEvent_t event_start_##traceName, event_stop_##traceName; \
HANDLE_CUDA_ERROR(cudaEventCreate(&event_start_##traceName)); \
HANDLE_CUDA_ERROR(cudaEventCreate(&event_stop_##traceName)); \
HANDLE_CUDA_ERROR(cudaEventRecord(event_start_##traceName, 0)); 

#define TIME_TRACE_CUDA_EVENT_STOP(traceName) \
float fElapsedTimeViaGPU_##traceName; \
HANDLE_CUDA_ERROR(cudaEventRecord(event_stop_##traceName, 0)); \
HANDLE_CUDA_ERROR(cudaEventSynchronize(event_stop_##traceName)); \
HANDLE_CUDA_ERROR(cudaEventElapsedTime(&fElapsedTimeViaGPU_##traceName, event_start_##traceName, event_stop_##traceName));\
printf("%s elapsed time via GPU is %f ms.\n", #traceName, fElapsedTimeViaGPU_##traceName); \
HANDLE_CUDA_ERROR(cudaEventDestroy(event_start_##traceName));\
HANDLE_CUDA_ERROR(cudaEventDestroy(event_stop_##traceName)); 
