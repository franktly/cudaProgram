#include "stdio.h"
#include <cuda_runtime.h>
#include "../CudaHelper.h"

const unsigned int BLOCK_SIZE = 32;
const unsigned int GRID_SIZE = 1;
const unsigned int SHARED_SIZE = 128;

struct myStruct
{
	/*
	 * float x, y, z, w;
	 */
	float arr[32];
	float padding;
};

__global__ void BankTest(unsigned long long *time)
{
	/*
	 * __shared__ float share[SHARED_SIZE];
	 */
	__shared__ myStruct share[SHARED_SIZE];
	unsigned long long beginTime = clock();

	// Situation 1: bank broadcast
	/*
	 * share[0]++;
	 */

	// Situation 2: no bank conflict, each thread in a warp access different bank(0~31)
	/*
	 * share[threadIdx.x]++;
	 */

	 /*
	  * share[threadIdx.x].x++;
	  */
	 share[threadIdx.x].arr[0]++;

	// Situation 3: two-way bank conflict, each thread in a warp access even banks(0, 2, 4... ,30), odd banks are not accessed
	/*
	 * share[threadIdx.x * 2]++; 
	 * share[threadIdx.x * 3]++
	 */

	// Situation 4: four-way bank conflict, each thread in a warp access even banks(0, 2, 4... ,30), odd banks are not accessed
	/*
	 * share[threadIdx.x * 4]++;
	 * share[threadIdx.x * 5]++;
	 */

	// Situation 5: 32-way bank conflict, each thread in a warp access only bank 0;
	 /*
	  * share[threadIdx.x * 32]++;
	  * share[threadIdx.x * 31]++;
	  */

	unsigned long long endTime = clock();
	*time = endTime - beginTime;
}

int main(int argc, char* argv[])
{
	unsigned long long  h_time;
	unsigned long long  *d_time;

	HANDLE_CUDA_ERROR(cudaMalloc((void**)&d_time, sizeof(unsigned long long)));

	//test for 10 times 
	for (int i = 0; i < 10; i++)
	{
		BankTest<<<GRID_SIZE, BLOCK_SIZE>>>(d_time);

		TIME_TRACE_CUDA_EVENT_START(cudaMemcpyDeviceToHost);
		HANDLE_CUDA_ERROR(cudaMemcpy(&h_time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
		TIME_TRACE_CUDA_EVENT_STOP(cudaMemcpyDeviceToHost);

		printf("bank test cost clocks: %d\n", (h_time - 14)/32);
	}

	HANDLE_CUDA_ERROR(cudaFree(d_time));

	// device reset to enable profile
	HANDLE_CUDA_ERROR(cudaDeviceReset());

	return 0;
}
