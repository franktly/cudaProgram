#include "stdio.h"
#include <cuda_runtime.h>

bool InitCuda(void)
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

	return true;
}

int main(int argv, char* argc[])
{
	if(!InitCuda())
	{
		printf("CUDA initialized failed!\n");
		return 0;
	}

	printf("CUDA initialized success!\n");

}
