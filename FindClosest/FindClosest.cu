#include "stdio.h"
#include <cuda_runtime.h>
#include "../CudaHelper.h"
#include <assert.h>
#include <sys/time.h>     // time
#include <stdlib.h>   // (s)rand
#include <math.h>    // sqrt


struct Float3
{
	float x;
	float y;
	float z;
};

const unsigned int COUNT = 1 << 12; // 4K

void FindClosestCPU(int* indexArray,Float3* pointArray, unsigned int count)
{
	assert(NULL != indexArray);
	assert(NULL != pointArray);
	assert(count >= 2);

	float closestDistance = 10000000.0f;
	float curPointDistance = 0.0f;
	for(int curPoint = 0; curPoint < count; curPoint++) 
	{
		for(int i = 0; i < count; i++)
		{
			if( i == curPoint)
			{
				continue;
			}

			curPointDistance = sqrt((pointArray[curPoint].x - pointArray[i].x) * (pointArray[curPoint].x - pointArray[i].x) +
					(pointArray[curPoint].y - pointArray[i].y) * (pointArray[curPoint].y - pointArray[i].y) +
					(pointArray[curPoint].z - pointArray[i].z) * (pointArray[curPoint].z - pointArray[i].z));

			/*
			 * curPointDistance = (pointArray[curPoint].x - pointArray[i].x) * (pointArray[curPoint].x - pointArray[i].x) +
			 *     (pointArray[curPoint].y - pointArray[i].y) * (pointArray[curPoint].y - pointArray[i].y) +
			 *     (pointArray[curPoint].z - pointArray[i].z) * (pointArray[curPoint].z - pointArray[i].z);
			 */

			if(curPointDistance < closestDistance)
			{
				closestDistance = curPointDistance;
				indexArray[curPoint] = i;
			}
		}

	}

}

int main(int argv, char* argc[])
{
	int *h_indexOfClosestPoint = new int[COUNT];
	assert(NULL != h_indexOfClosestPoint);
	Float3 *h_pointArray       = new Float3[COUNT];
	assert(NULL != h_pointArray);

	srand((int)time(NULL));
	for(int i = 0; i < COUNT; i++)
	{
		h_pointArray[i].x = (float)(rand()%1000);
		h_pointArray[i].y = (float)(rand()%1000);
		h_pointArray[i].z = (float)(rand()%1000);
	}

	struct timeval start_time;
	struct timeval stop_time;
	for(int i = 0; i < 10; i++)
	{
		gettimeofday(&start_time, 0); 

		FindClosestCPU(h_indexOfClosestPoint, h_pointArray, COUNT);

		gettimeofday(&stop_time, 0); 
		float fElapsedTimeViaCPU =  1000.0 * (stop_time.tv_sec - start_time.tv_sec) + (0.001 * (stop_time.tv_usec - start_time.tv_usec)); 
		printf("%d elapsed time via CPU is %f ms.\n", i , fElapsedTimeViaCPU); 
	}


	// print first 0~10 index point
	for (int i = 0; i< 10; i++)
	{
		printf("index [%d](point is [%f,%f,%f]) is closest to index [%d](point is [%f,%f,%f])\n",
				i, h_pointArray[i].x, h_pointArray[i].y, h_pointArray[i].z,
				h_indexOfClosestPoint[i], h_pointArray[h_indexOfClosestPoint[i]].x, h_pointArray[h_indexOfClosestPoint[i]].y,
				h_pointArray[h_indexOfClosestPoint[i]].z);
	}

	if(NULL != h_indexOfClosestPoint) { delete []h_indexOfClosestPoint;}
	if(NULL != h_pointArray) { delete []h_pointArray;}

}


