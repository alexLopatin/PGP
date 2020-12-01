#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <algorithm>

__global__ void Histohram(int* devCount, int* arr, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;

	for (int i = idx; i < size; i+= offsetx)
	{
		atomicAdd(&devCount[arr[i]], 1);
	}
}

__global__ void CountSort(int* devScan, int* arr, int* out, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;

	for (int i = idx; i < size; i += offsetx)
	{
		out[atomicAdd(&devScan[arr[i]], -1) - 1] = arr[i];
	}
}

const int BLOCK_SIZE = 256;

__global__ void KernelBlockScan(int* devArr, int* newDevArr)
{
	int blockSize = blockDim.x;
	__shared__ int arr[BLOCK_SIZE];

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	arr[threadIdx.x] = devArr[idx];
	__syncthreads();

	int d = 1;

	while (d < blockSize)
	{
		if (2 * d + threadIdx.x * 2 * d - 1 < blockSize)
			arr[2 * d + threadIdx.x * 2 * d - 1] += arr[d + threadIdx.x * 2 * d - 1];
		d *= 2;
		__syncthreads();
	}

	int last = 0;

	if (threadIdx.x == blockSize - 1)
	{
		last = arr[threadIdx.x];
		arr[threadIdx.x] = 0;
	}

	d /= 2;

	__syncthreads();
	
	while (d >= 1)
	{
		if (d * 2 * threadIdx.x + 2 * d - 1 < blockSize)
		{
			auto t = arr[d * 2 * threadIdx.x + d - 1];
			arr[d * 2 * threadIdx.x + d - 1] = arr[d * 2 * threadIdx.x + 2 * d - 1];
			arr[d * 2 * threadIdx.x + 2 * d - 1] += t;
		}

		d /= 2;
		__syncthreads();
	}

	if (threadIdx.x == blockSize - 1)
	{
		devArr[idx] = last;
		newDevArr[blockIdx.x] = last;
	}
	else
	{
		devArr[idx] = arr[threadIdx.x + 1];
	}
}

__global__ void KernelBlockShift(int* devArr, int* newArr)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (blockIdx.x > 0)
		devArr[idx] += newArr[blockIdx.x - 1];
}

int Max(int a, int b)
{
	return a > b
		? a
		: b;
}

int Min(int a, int b)
{
	return a < b
		? a
		: b;
}

void Scan(int* devCount, int size)
{
	//auto newHostCount = &hostCount[Ceil(size, BLOCK_SIZE) * BLOCK_SIZE];
	//auto newHostCount = new int[Ceil(size, BLOCK_SIZE)];

	int blockCount = Max(1, size / BLOCK_SIZE);
	int blockSize = Min(size, BLOCK_SIZE);
	int* newDevCount;
	cudaMalloc(&newDevCount, sizeof(int) * blockSize);
	
	KernelBlockScan<<< blockCount, blockSize >>>(devCount, newDevCount);

	if (size > BLOCK_SIZE)
	{
		Scan(newDevCount, size / BLOCK_SIZE);
		KernelBlockShift<<<size / BLOCK_SIZE, BLOCK_SIZE >>>(devCount, newDevCount);
	}

	cudaFree(newDevCount);
}

using namespace std;

const int MAX_NUMBER = 16777215;

int main(int argc, const char** argv)
{
	int size;
	cin >> size;
	//fread(&size, sizeof(int), 1, stdin);

	auto hostArray = new int[size];
	//fread(hostArray, sizeof(int), size, stdin);

	fprintf(stderr, "size = %d\n", size);
	for (int i = 0; i < size; i++)
	{
		cin >> hostArray[i];
		//fprintf(stderr, "%d ", hostArray[i]);
	}

	int* devCount;
	cudaMalloc(&devCount, sizeof(int) * (MAX_NUMBER + 1));
	cudaMemset(&devCount, 0, sizeof(int) * (MAX_NUMBER + 1));

	int* devArray;
	cudaMalloc(&devArray, sizeof(int) * size);
	cudaMemcpy(devArray, hostArray, sizeof(int) * size, cudaMemcpyHostToDevice);

	Histohram<<<256, 256>>>(devCount, devArray, size);
	Scan(devCount, MAX_NUMBER + 1);

	int* outDevArray;
	cudaMalloc(&outDevArray, sizeof(int) * size);
	CountSort<<<256, 256>>>(devCount, devArray, outDevArray, size);

	cudaMemcpy(hostArray, outDevArray, sizeof(int) * size, cudaMemcpyDeviceToHost);

	//fwrite(hostArray, sizeof(int), size, stdout);

	fprintf(stderr, "output:\n");

	for (int i = 0; i < size; i++)
	{
		fprintf(stderr, "%d ", hostArray[i]);
	}
}