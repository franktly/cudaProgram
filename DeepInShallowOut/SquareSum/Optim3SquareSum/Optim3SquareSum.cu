#include "stdio.h"
#include <cuda_runtime.h>
#include "../../../CudaHelper.h"

const unsigned int  DATE_SIZE  = 1 << 24; // 8M 
const unsigned int  BLOCK_SIZE = 1024; // block size
/*
 * 
 * Many Block: every thread exec 1 computation task and all computation task is exec in device 
  */
// Kernel function to compute square sum of an int array to a result 
__global__ void SquareSum(int *pInputData, int *pResult)
{
	const int tid = threadIdx.x +blockDim.x * blockIdx.x;
	if(tid < DATE_SIZE)
	{
		*pResult += pInputData[tid] * pInputData[tid];
	}
}


int main(int argv, char* argc[])
{
	// Get cuda device count
	int iCount;
	cudaGetDeviceCount(&iCount);
	if(0 == iCount)
	{
		printf("There is no cuda device\n");
		return false; 
	}

	// Find the first suitable device
	int i;
	for (i = 0; i < iCount; i++)
	{
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{
			// find a prop > CUDA 1.X device and break
			if(prop.major >= 1)
			{
				break;
			}
		}
	}

	// can not find a prop > CUDA 1.X device and return false
	if(i == iCount)
	{
		printf("There is no CUDA 1.X device\n");
		return false;
	}

	// Set the suitable device to current
	cudaSetDevice(i);

	// Malloc host data
	int *pHostData = (int*)malloc(sizeof(int)*DATE_SIZE);
	int  hostResult = 0;
	if( 0 == pHostData)
	{
		printf("malloc host data failed!!!\n");
		return -1;
	}

	// Generate 16M rand data range from 0 to 4
	for(int i = 0; i < DATE_SIZE; i++)
	{
		pHostData[i] = rand() % 5;
	}

	// Malloc device data
	int *pDeviceData = NULL;
	int *pDeviceResult = NULL;
	
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&pDeviceData, sizeof(int) * DATE_SIZE));
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&pDeviceResult, sizeof(int)));

	printf("\nGPU COMPUTE BEGIN********************\n");

	// Record total time elapsed via GPU
	TIME_TRACE_CUDA_EVENT_START(TotalElpasedTimeViaGPU);

	// Copy host data to device
	TIME_TRACE_CUDA_EVENT_START(cudaMemcpyHostToDevice);
	HANDLE_CUDA_ERROR(cudaMemcpy(pDeviceData, pHostData, sizeof(int) * DATE_SIZE, cudaMemcpyHostToDevice));
	TIME_TRACE_CUDA_EVENT_STOP(cudaMemcpyHostToDevice);

	// Execute Kernel 
	TIME_TRACE_CUDA_EVENT_START(SqureSumKernel);
	unsigned int GRID_SIZE = (DATE_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE;
	SquareSum<<<GRID_SIZE, BLOCK_SIZE>>>(pDeviceData, pDeviceResult);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(err));
	}
	TIME_TRACE_CUDA_EVENT_STOP(SqureSumKernel);

	// Copy result from device
	TIME_TRACE_CUDA_EVENT_START(cudaMemcpyDeviceToHost);
	HANDLE_CUDA_ERROR(cudaMemcpy(&hostResult, pDeviceResult, sizeof(int), cudaMemcpyDeviceToHost)); 
	TIME_TRACE_CUDA_EVENT_STOP(cudaMemcpyDeviceToHost);
	
	TIME_TRACE_CUDA_EVENT_STOP(TotalElpasedTimeViaGPU);

	// Free device memory
	HANDLE_CUDA_ERROR(cudaFree(pDeviceData));
	HANDLE_CUDA_ERROR(cudaFree(pDeviceResult));

	// Print result
	printf("Square Sum Computed Via Result GPU is %d.\n", hostResult);

	// cudaDeviceReset to ensure Visual Profile run correctly
	HANDLE_CUDA_ERROR(cudaDeviceReset());

	printf("\nGPU COMPUTE END********************\n");

	printf("\nCPU COMPUTE BEGIN********************\n");
	// Compute in CPU for comparision
	hostResult = 0;

	TIME_TRACE_CPU_START(TotalElpasedTimeViaCPU);
	for (int i = 0 ; i < DATE_SIZE; i++)
	{
		hostResult += pHostData[i] * pHostData[i];
	}
	TIME_TRACE_CPU_STOP(TotalElpasedTimeViaCPU);

	// Free host memory
	free(pHostData); pHostData = NULL;

	// Print result
	printf("Square Sum Computed Result Via CPU is %d.\n", hostResult);

	printf("\nCPU COMPUTE END********************\n");

	return 0;
	
}
