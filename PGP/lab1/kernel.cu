#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

void ReadArray(double* arr, int size);
void WriteArray(double* arr, int size);

void CudaOperation(double* first, double* second, double* result, int vectorLength);
__global__ void OperateParallel(double* firstDevice, double* secondDevice, double* resultDevice, int vectorLength);

__device__ double Min(double a, double b);

int main()
{
	int vectorLength;
	std::cin >> vectorLength;

	double* first;
	double* second;
	double* result;

	first = (double*)malloc(sizeof(double) * vectorLength);
	second = (double*)malloc(sizeof(double) * vectorLength);
	result = (double*)malloc(sizeof(double) * vectorLength);

	ReadArray(first, vectorLength);
	ReadArray(second, vectorLength);

	CudaOperation(first, second, result, vectorLength);

	WriteArray(result, vectorLength);
}

void ReadArray(double* arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		std::cin >> arr[i];
	}
}

void WriteArray(double* arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		std::cout << arr[i];

		if (i < size - 1)
		{
			std::cout << ' ';
		}
	}
}

void CudaOperation(double* first, double* second, double* result, int vectorLength)
{
	double* firstDevice = 0;
	double* secondDevice = 0;
	double* resultDevice = 0;

	cudaSetDevice(0);

	cudaMalloc((void**)&firstDevice, vectorLength * sizeof(double));
	cudaMalloc((void**)&secondDevice, vectorLength * sizeof(double));
	cudaMalloc((void**)&resultDevice, vectorLength * sizeof(double));

	cudaMemcpy(firstDevice, first, vectorLength * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(secondDevice, second, vectorLength * sizeof(double), cudaMemcpyHostToDevice);

	OperateParallel << <1, 256 >> > (firstDevice, secondDevice, resultDevice, vectorLength);
	cudaDeviceSynchronize();

	cudaMemcpy(result, resultDevice, vectorLength * sizeof(double), cudaMemcpyDeviceToHost);
}

__device__ int Min(int a, int b)
{
	return (a > b)
		? b
		: a;
}

__device__ int Max(int a, int b)
{
	return (a < b)
		? b
		: a;
}

__global__ void OperateParallel(double* firstDevice, double* secondDevice, double* resultDevice, int vectorLength)
{
	int length = Max(vectorLength / blockDim.x, 1);
	int left = Min(threadIdx.x * length, vectorLength);
	int right = (threadIdx.x == blockDim.x - 1)
		? vectorLength - 1
		: Min(left + length - 1, vectorLength - 1);

	for (int i = left; i <= right; i++)
	{
		resultDevice[i] = Min(firstDevice[i], secondDevice[i]);
	}
}

__device__ double Min(double a, double b)
{
	return (a > b)
		? b
		: a;
}