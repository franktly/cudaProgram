#include "stdio.h"
#include <cuda_runtime.h>
#include "../../CudaHelper.h"

const unsigned int  BLOCK_SIZE_X = 16; // block x size
const unsigned int  BLOCK_SIZE_Y = 16; // block y size
const unsigned int  GRID_SIZE_X  = 4096/16; // grid x size
const unsigned int  GRID_SIZE_Y  = 4096/16; // grid y size
const unsigned int  IMAGE_WIDTH  = 4096; // image width size
const unsigned int  IMAGE_HEIGHT = 4096; // image height size
const unsigned int  KERNEL_SIZE = 5;    // Gaussian Kernel Size

unsigned char* GenerateRandGrayPics(unsigned int width, unsigned int height) 
{
	unsigned char* pImg = (unsigned char*)malloc(width * height * sizeof(unsigned char));
	if(NULL == pImg) { printf("malloc img buffer failed!!!\n"); return NULL;} 

	for(int i = 0; i < height; i++) 
	{
		for(int j = 0; j < width; j++) 
		{
			pImg[i * width + j] = rand()%256;  // 0~255 
		}
	}

	return pImg;
}

bool CompareGrayImg(unsigned char* pImgA, unsigned char* pImgB, unsigned int width, unsigned height)
{
	bool ret = true;

	if((NULL == pImgA)||(NULL == pImgB))
	{
		printf("input img is empty!!!\n");
		return false;
	}
	
	for(int i = 0; i < height; i++) 
	{
		for(int j = 0; j < width; j++) 
		{
			if(pImgA[i * width + j] != pImgB[i * width + j])
			{
				printf("img[%d][%d] gray value is not the same, pImgA is %d, pImgB is %d\n ",
						i, j, pImgA[i * width + j], pImgB[i * width + j]);
				ret = false;
			}
		}
	}

	if(ret)
	{
		printf("Compare 2D Gray Img Passed!!!\n");
	}
	else
	{
		printf("Compare 2D Gray Img Failed!!!\n");
	}

	return ret;
}


void GaussianSmooth_CPU(unsigned char* pInputImg, unsigned char *pOutputImg, unsigned int width, unsigned height)
{
	int gs_kernel[KERNEL_SIZE][KERNEL_SIZE] =
	{
		{1, 4, 7, 4, 1},
		{4, 16, 26, 16, 4},
		{7, 26, 41, 26, 7},
		{4, 16, 26, 16, 4},
		{1, 4, 7, 4, 1}
	};  // kernel sum is 273

	// loop in every pixel(height * width)
	for (int row = 0; row < height; row++)
	{
		for(int col = 0; col < width; col++)
		{
			int sum = 0;
			int out_row = row;
			int out_col = col;

			// loop in every kernel(KERNEL_SIZE * KERNEL_SIZE), for pixel img begin from　-KERNEL_SIZE/2;
			// padding pixel value is the edge pixel value
			for( int i = -KERNEL_SIZE/2; i < KERNEL_SIZE/2; i++)
			{
				for( int j = -KERNEL_SIZE/2; j < KERNEL_SIZE/2; j++)
				{
					row = row + i;
					col = col + j;
					row = min(max(0, row), width -1);
					col = min(max(0, col), height -1);
					unsigned char tmpPixel = *(pInputImg + width * row + col);
					sum += tmpPixel * gs_kernel[i+KERNEL_SIZE/2][j+KERNEL_SIZE/2];
				}
			}

			int final_pixel = sum/273;

			if(final_pixel < 0)
			{
				final_pixel = 0;
			}
			else if(final_pixel > 255)
			{
				final_pixel = 255;
			}

			*(pOutputImg + out_row * width + out_col) = final_pixel;
		}
	}
}

// Kernel function to compute square sum of an int array to a result 
__global__ void GaussianSmooth_Kernel(unsigned char *pInputImg, unsigned char *pOutputImg, int* width, int* height)
{
	int row = threadIdx.x +blockDim.x * blockIdx.x;
	int col = threadIdx.y +blockDim.y * blockIdx.y;

	int gs_kernel[KERNEL_SIZE][KERNEL_SIZE] =
	{
		{1, 4, 7, 4, 1},
		{4, 16, 26, 16, 4},
		{7, 26, 41, 26, 7},
		{4, 16, 26, 16, 4},
		{1, 4, 7, 4, 1}
	};  // kernel sum is 273


	int sum = 0;
	int out_row = row;
	int out_col = col;

	// loop in every kernel(KERNEL_SIZE * KERNEL_SIZE), for pixel img begin from　-KERNEL_SIZE/2;
	// padding pixel value is the edge pixel value
	for( int i = -KERNEL_SIZE/2; i < KERNEL_SIZE/2; i++)
	{
		for( int j = -KERNEL_SIZE/2; j < KERNEL_SIZE/2; j++)
		{
			row = row + i;
			col = col + j;
			row = min(max(0, row), *width -1);
			col = min(max(0, col), *height -1);
			unsigned char tmpPixel = *(pInputImg + *width * row + col);
			sum += tmpPixel * gs_kernel[i+KERNEL_SIZE/2][j+KERNEL_SIZE/2];
		}
	}

	int final_pixel = sum/273;

	if(final_pixel < 0)
	{
		final_pixel = 0;
	}
	else if(final_pixel > 255)
	{
		final_pixel = 255;
	}

	*(pOutputImg + out_row * (*width) + out_col) = final_pixel;
}

int main(int argv, char* argc[])
{
	// deal with input param
	int blockSizeX = BLOCK_SIZE_X;
	int blockSizeY = BLOCK_SIZE_Y;
	int gridSizeX  = GRID_SIZE_X;
	int gridSizeY  = GRID_SIZE_Y;
	int width      = IMAGE_WIDTH;
	int height     = IMAGE_HEIGHT;

	if(argv > 1)
	{
		blockSizeX = atoi(argc[1]);
	}
	if(argv > 2)
	{
		blockSizeY = atoi(argc[2]);
	}
	if(argv > 3)
	{
		gridSizeX = atoi(argc[3]);
	}
	if(argv > 4)
	{
		gridSizeY = atoi(argc[4]);
	}
	if(argv > 5)
	{
		width = atoi(argc[5]);
	}
	if(argv > 6)
	{
		height = atoi(argc[6]);
	}

	printf("blockSizeX is %d\n", blockSizeX);
	printf("blockSizeY is %d\n", blockSizeY);
	printf("gridSizeX is %d\n", gridSizeX);
	printf("gridSizeY is %d\n", gridSizeY);
	printf("width is %d\n", width);
	printf("height is %d\n", height);

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
	unsigned char *pHostInputImg  = GenerateRandGrayPics(width, height);
	if(NULL == pHostInputImg) { printf("malloc host input img buffer failed!!!\n"); return -1;} 
	unsigned char* pHostOutputImg = (unsigned char*)malloc(width * height * sizeof(unsigned char));
	if(NULL == pHostOutputImg) { printf("malloc host output img buffer failed!!!\n"); return -1;} 

	// Malloc device data
	unsigned char *pDeviceInputImg  = NULL;
	unsigned char *pDeviceOutputImg = NULL;
	int           *pDeviceImageWidth  = NULL;
	int           *pDeviceImageHeight = NULL;
	
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&pDeviceInputImg,  sizeof(unsigned char) * width * height));
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&pDeviceOutputImg, sizeof(unsigned char) * width * height));
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&pDeviceImageWidth, sizeof(int)));
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&pDeviceImageHeight, sizeof(int)));

	printf("\nGPU COMPUTE BEGIN********************\n");

	// Record total time elapsed via GPU
	TIME_TRACE_CUDA_EVENT_START(TotalElpasedTimeViaGPU);

	// Copy host data to device
	TIME_TRACE_CUDA_EVENT_START(cudaMemcpyHostToDevice);
	HANDLE_CUDA_ERROR(cudaMemcpy(pDeviceInputImg, pHostInputImg, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice));
	HANDLE_CUDA_ERROR(cudaMemcpy(pDeviceImageWidth,  &width,  sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_CUDA_ERROR(cudaMemcpy(pDeviceImageHeight, &height, sizeof(int), cudaMemcpyHostToDevice));
	TIME_TRACE_CUDA_EVENT_STOP(cudaMemcpyHostToDevice);

	// Execute Kernel 
	TIME_TRACE_CUDA_EVENT_START(GaussianSmoothKernel);
	// Set the minium of const GRID_SIZE and computation result of grid size based on image input size to the final grid size
	int grid_size_x = min((width  + blockSizeX-1)/blockSizeX, gridSizeX);
	int grid_size_y = min((height + blockSizeY-1)/blockSizeY, gridSizeY);
	dim3 block(blockSizeX, blockSizeY, 1);
	dim3 grid(grid_size_x, grid_size_y, 1);
	GaussianSmooth_Kernel<<<grid, block>>>(pDeviceInputImg, pDeviceOutputImg, pDeviceImageWidth, pDeviceImageHeight);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(err));
	}
	TIME_TRACE_CUDA_EVENT_STOP(GaussianSmoothKernel);

	// Copy result from device
	TIME_TRACE_CUDA_EVENT_START(cudaMemcpyDeviceToHost);
	HANDLE_CUDA_ERROR(cudaMemcpy(pHostOutputImg, pDeviceOutputImg, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost)); 
	TIME_TRACE_CUDA_EVENT_STOP(cudaMemcpyDeviceToHost);
	
	TIME_TRACE_CUDA_EVENT_STOP(TotalElpasedTimeViaGPU);

	// Free device memory
	HANDLE_CUDA_ERROR(cudaFree(pDeviceInputImg));
	HANDLE_CUDA_ERROR(cudaFree(pDeviceOutputImg));
	HANDLE_CUDA_ERROR(cudaFree(pDeviceImageWidth));
	HANDLE_CUDA_ERROR(cudaFree(pDeviceImageHeight));

	// cudaDeviceReset to ensure Visual Profile run correctly
	HANDLE_CUDA_ERROR(cudaDeviceReset());

	printf("GPU COMPUTE END********************\n");

	printf("\nCPU COMPUTE BEGIN********************\n");
	// Compute in CPU for comparision

	unsigned char* pHostOutputImg_CPU = (unsigned char*)malloc(width * height * sizeof(unsigned char));
	if(NULL == pHostOutputImg_CPU) { printf("malloc host output img buffer failed!!!\n"); return -1;} 

	TIME_TRACE_CPU_START(TotalElpasedTimeViaCPU);
	GaussianSmooth_CPU(pHostInputImg, pHostOutputImg_CPU, width, height);
	TIME_TRACE_CPU_STOP(TotalElpasedTimeViaCPU);

	printf("CPU COMPUTE END********************\n");

	//Print Compare Compute Result 
	CompareGrayImg(pHostOutputImg,pHostOutputImg_CPU, width, height);

	// Free host memory
	free(pHostInputImg); pHostInputImg = NULL;
	free(pHostOutputImg); pHostOutputImg = NULL;
	free(pHostOutputImg_CPU); pHostOutputImg_CPU = NULL;

	return 0;
}
