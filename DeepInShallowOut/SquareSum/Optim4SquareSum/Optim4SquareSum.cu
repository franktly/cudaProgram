#include "stdio.h"
#include <cuda_runtime.h>
#include "../../../CudaHelper.h"

const unsigned int  DATE_SIZE  = 1 << 24; // 16M 
const unsigned int  BLOCK_SIZE = 1024; // block size
const unsigned int  GRID_SIZE  = 8; // grid size
/*
 * 
 * Many Block: every thread(totally thread number is BLOCK_SIZE*GRID_SIZE) exec DATE_SIZE/BLOCK_SIZE*GRID_SIZE computation task 
 * And Use Shared Memory 
 * if BLOCK_SIZE*GRID_SIZE == DATE_SIZE, every thread exec 1 time)
 * 
 * friendly for global memory access(data space locality and benefit for cache line), adjacent thread access adjacent data addr space
 * thread k compute column k data:(k = 0 ~ BLOCK_SIZE*GRID_SIZE-1) 
 * 
 * BlockId :                               bid0                                          |...     bidGRID_SIZE-1
 * --------------------------------------------------------------------------------------|----------------
 * ThreadId:     tid0                                tid1                 tidBLOCK_SIZE-1|...  tidBLOCK_SIZE*GRID_SIZE-1
 * --------------------------------------------------------------------------------------|----------------
 * DataId  :     dat0                                dat1                 datBLOCK_SIZE-1|...  datBLOCK_SIZE*GRID_SIZE-1
 * DataId  :     datBLOCK_SIZE*GRID_SIZE+0    datBLOCK_SIZE*GRID_SIZE+1   datBLOCK_SIZE-1|...  datBLOCK_SIZE*GRID_SIZE+BLOCK_SIZE*GRID_SIZE-1
 * DataId  :     datBLOCK_SIZE*GRID_SIZE*2+0  datBLOCK_SIZE*GRID_SIZE*2+1 datBLOCK_SIZE-1|...  datBLOCK_SIZE*GRID_SIZE*2+BLOCK_SIZE*GRID_SIZE-1
 * 
 * ...
 * Shared  :   shared[0]                             shared[1]    shared[tidBLOCK_SIZE-1]|... shared[0] ...  shared[tidBLOCK_SIZE-1]  
 * --------------------------------------------------------------------------------------|----------------
 *SharedAdd:   shared[0]                                                                 |... shared[0]
 * --------------------------------------------------------------------------------------|----------------
  */
// Kernel function to compute square sum of an int array to a result 
__global__ void SquareSum(int *pInputData, int *pResult)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int gtid = threadIdx.x +blockDim.x * blockIdx.x;
	int i = 0;
	__shared__ int shared[BLOCK_SIZE];
	shared[tid] = 0;

	// each different block has its own different shared memory result
	// in each different block, shared[tid] save each thread[tid] added result (the result of the specified thread in a block exec many times computation task) in specified block[bid]
	for(i = gtid; i < DATE_SIZE; i = i + BLOCK_SIZE * GRID_SIZE)
	{
		shared[tid] += pInputData[i] * pInputData[i];
	}

	// sync threads in a block
	__syncthreads();

	// add each thread's shared memory value(the result of the specified thread in the specified block[bid] exec many times computation task) in the specified block[bid] computation result and save result to shared[0] And pResult[bid]

	//① ONLY tid=0 participates in block shared memory addition computation task and is belongs to serial computation 
/*
 *     if(tid == 0)
 *     {
 *         for(i = 1; i < BLOCK_SIZE; i++)
 *         {
 *             shared[0] += shared[i];
 *         }
 *         pResult[bid] = shared[0];  // every block saves (all threads in the block exec many times) the finial computatiom result to  pResult[bid]
 *     }
 * 
 */

	//② Parallize block shared memory addition computation, (TREE ADD)
	int offset = BLOCK_SIZE/2;
	while(offset > 0)
	{
		if(tid < offset)
		{
			shared[tid] += shared[tid + offset];
		}
		offset >>=1;
		// sync threads in a block
		__syncthreads();
	}
	pResult[bid] = shared[0];

	//③ Parallize block shared memory addition computation, (SPREAD TREE ADD)

/*
 *     int offset = BLOCK_SIZE/2;
 *     pResult[bid] = shared[0];
 *     if(tid < 512) { shared[tid] += shared[tid+512];}
 *     __syncthreads();
 *     if(tid < 256) { shared[tid] += shared[tid+256];}
 *     __syncthreads();
 *     if(tid < 128) { shared[tid] += shared[tid+128];}
 *     __syncthreads();
 *     if(tid < 64) { shared[tid] += shared[tid+64];}
 *     __syncthreads();
 *     if(tid < 32) { shared[tid] += shared[tid+32];}
 *     __syncthreads();
 *     if(tid < 16) { shared[tid] += shared[tid+16];}
 *     __syncthreads();
 *     if(tid < 8) { shared[tid] += shared[tid+8];}
 *     __syncthreads();
 *     if(tid < 4) { shared[tid] += shared[tid+4];}
 *     __syncthreads();
 *     if(tid < 2) { shared[tid] += shared[tid+2];}
 *     __syncthreads();
 *     if(tid < 1) { shared[tid] += shared[tid+1];}
 *     __syncthreads();
 * 
 *     pResult[bid] = shared[0];
 */
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
	int *pHostBlockData = (int*)malloc(sizeof(int) * GRID_SIZE);
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
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&pDeviceResult, sizeof(int) * GRID_SIZE));

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
	HANDLE_CUDA_ERROR(cudaMemcpy(pHostBlockData, pDeviceResult, sizeof(int) * GRID_SIZE, cudaMemcpyDeviceToHost)); 
	TIME_TRACE_CUDA_EVENT_STOP(cudaMemcpyDeviceToHost);
	
	TIME_TRACE_CUDA_EVENT_STOP(TotalElpasedTimeViaGPU);

	// Free device memory
	HANDLE_CUDA_ERROR(cudaFree(pDeviceData));
	HANDLE_CUDA_ERROR(cudaFree(pDeviceResult));

	// Add every thread result in CPU
	TIME_TRACE_CPU_START(AddEveryThreadData);
	for (int i = 0 ; i < GRID_SIZE; i++)
	{
		hostResult += pHostBlockData[i];
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
	free(pHostBlockData); pHostBlockData = NULL;
	free(pHostData); pHostData = NULL;

	// Print result
	printf("Square Sum Computed Result Via CPU is %d.\n", hostResult);

	printf("\nCPU COMPUTE END********************\n");

	return 0;
	
}
