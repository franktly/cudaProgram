#include "stdio.h"
#include <cuda_runtime.h>
#include "../../../CudaHelper.h"

const unsigned int  DATE_SIZE  = 1 << 24; // 16M 
const unsigned int  BLOCK_SIZE = 1024; // block size
const unsigned int  GRID_SIZE  = 8; // grid size
/*
 * 
 * Many Block: every thread(totally thread number is BLOCK_SIZE*GRID_SIZE) exec DATE_SIZE/BLOCK_SIZE*GRID_SIZE computation task
 * if BLOCK_SIZE*GRID_SIZE == DATE_SIZE, every thread exec 1 time)
 * 
 * friendly for global memory access(data space locality and benefit for cache line), adjacent thread access adjacent data addr space
 * thread k compute column k data:(k = 0 ~ BLOCK_SIZE*GRID_SIZE-1) 
 * 
 * ThreadId:     tid0                                tid1                 ...  tidBLOCK_SIZE*GRID_SIZE-1
 * -------------------------------------------------------------------------------------------------------
 * DataId  :     dat0                                dat1                 ...  datBLOCK_SIZE*GRID_SIZE-1
 * DataId  :     datBLOCK_SIZE*GRID_SIZE+0    datBLOCK_SIZE*GRID_SIZE+1   ...  datBLOCK_SIZE*GRID_SIZE+BLOCK_SIZE*GRID_SIZE-1
 * DataId  :     datBLOCK_SIZE*GRID_SIZE*2+0  datBLOCK_SIZE*GRID_SIZE*2+1 ...  datBLOCK_SIZE*GRID_SIZE*2+BLOCK_SIZE*GRID_SIZE-1
 * 
 * ...
 * 
 * badly for global memory access(data space locality and benefit for cache line), adjacent thread does not access adjacent data addr space
 * thread k compute row k data:(k = 0 ~ BLOCK_SIZE*GRID_SIZE-1)
 *																		   					                                ThreadId:
 * ---------------------------------------------------------------------------------------------------------------------------------
 * DataId  :     dat0                                dat1                  ...  datBLOCK_SIZE*GRID_SIZE-1                        tid0
 * DataId  :     datBLOCK_SIZE*GRID_SIZE+0    datBLOCK_SIZE*GRID_SIZE+1    ...  datBLOCK_SIZE*GRID_SIZE+BLOCK_SIZE*GRID_SIZE-1   tid1
 * DataId  :     datBLOCK_SIZE*GRID_SIZE*2+0  datBLOCK_SIZE*GRID_SIZE*2+   ...  datBLOCK_SIZE*GRID_SIZE*2+BLOCK_SIZE*GRID_SIZE-1 tid2
 * 
 * ... 
 */
// Kernel function to compute square sum of an int array to a result 
__global__ void SquareSum(int *pInputData, int *pResult)
{
	const int tid = threadIdx.x +blockDim.x * blockIdx.x;
	int i = 0;
	int result = 0;

	//  friendly for global memory access(data space locality and benefit for cache line), adjacent thread access adjacent data addr space
	for(i = tid; i < DATE_SIZE; i = i + BLOCK_SIZE * GRID_SIZE)
	{
		result += pInputData[i] * pInputData[i];
	}

	//  badly for global memory access(data space locality and benefit for cache line), adjacent thread does not access adjacent data addr space
	/*
	 * const int count = DATE_SIZE /(BLOCK_SIZE * GRID_SIZE);
	 * for( i = tid * count; i < (tid+1) * count; i++)
	 * {
	 *     result += pInputData[i] * pInputData[i];
	 * }
	 */

	pResult[tid] = result;
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
	int *pHostThreadData = (int*)malloc(sizeof(int)*BLOCK_SIZE * GRID_SIZE);
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
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&pDeviceResult, sizeof(int) * BLOCK_SIZE * GRID_SIZE));

	printf("\nGPU COMPUTE BEGIN********************\n");

	// Record total time elapsed via GPU
	TIME_TRACE_CUDA_EVENT_START(TotalElpasedTimeViaGPU);

	// Copy host data to device
	TIME_TRACE_CUDA_EVENT_START(cudaMemcpyHostToDevice);
	HANDLE_CUDA_ERROR(cudaMemcpy(pDeviceData, pHostData, sizeof(int) * DATE_SIZE, cudaMemcpyHostToDevice));
	TIME_TRACE_CUDA_EVENT_STOP(cudaMemcpyHostToDevice);

	// Execute Kernel 
	TIME_TRACE_CUDA_EVENT_START(SqureSumKernel);
	SquareSum<<<GRID_SIZE, BLOCK_SIZE>>>(pDeviceData, pDeviceResult);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(err));
	}
	TIME_TRACE_CUDA_EVENT_STOP(SqureSumKernel);

	// Copy result from device
	TIME_TRACE_CUDA_EVENT_START(cudaMemcpyDeviceToHost);
	HANDLE_CUDA_ERROR(cudaMemcpy(pHostThreadData, pDeviceResult, sizeof(int) * BLOCK_SIZE * GRID_SIZE, cudaMemcpyDeviceToHost)); 
	TIME_TRACE_CUDA_EVENT_STOP(cudaMemcpyDeviceToHost);
	
	TIME_TRACE_CUDA_EVENT_STOP(TotalElpasedTimeViaGPU);

	// Free device memory
	HANDLE_CUDA_ERROR(cudaFree(pDeviceData));
	HANDLE_CUDA_ERROR(cudaFree(pDeviceResult));

	// Add every thread result in CPU
	TIME_TRACE_CPU_START(AddEveryThreadData);
	for (int i = 0 ; i < BLOCK_SIZE * GRID_SIZE; i++)
	{
		hostResult += pHostThreadData[i];
	}
	TIME_TRACE_CPU_STOP(AddEveryThreadData);

	// Print result
	printf("Square Sum Computed Via Result GPU & CPU is %d.\n", hostResult);

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
	free(pHostThreadData); pHostThreadData = NULL;
	free(pHostData); pHostData = NULL;

	// Print result
	printf("Square Sum Computed Result Via CPU is %d.\n", hostResult);

	printf("\nCPU COMPUTE END********************\n");

	return 0;
	
}
