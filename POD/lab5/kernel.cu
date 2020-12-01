#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <algorithm>

#define CSC(call)                   \
do {                                \
    cudaError_t res = call;         \
    if (res != cudaSuccess) {       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                    \
    }                               \
} while(0)

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

const int BLOCK_SIZE = 1024;

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
	cudaMalloc((void**)&newDevCount, sizeof(int) * blockCount);
	
	KernelBlockScan<<< blockCount, blockSize >>>(devCount, newDevCount);
	cudaDeviceSynchronize();
	fprintf(stderr, "<<<%d, %d>>>\n", blockCount, blockSize);
	if (size > BLOCK_SIZE)
	{
		Scan(newDevCount, size / BLOCK_SIZE);
		KernelBlockShift<<<size / BLOCK_SIZE, BLOCK_SIZE >>>(devCount, newDevCount);
		cudaDeviceSynchronize();
	}

	cudaFree(newDevCount);
}

using namespace std;

const int MAX_NUMBER = 16777215;

__global__ void TestAdd(int* devInt)
{
	atomicAdd(&devInt[0], 1);
}

int main(int argc, const char** argv)
{
	//cudaSetDevice(1);

	/*auto testArr = new int[MAX_NUMBER + 1];

	for(int i = 0; i < MAX_NUMBER + 1; i++)
		testArr[i] = 1;
	int* devTestArr;

	cudaMalloc((void**)&devTestArr, sizeof(int) * (MAX_NUMBER + 1));

	cudaMemcpy(devTestArr, testArr, sizeof(int) * (MAX_NUMBER + 1), cudaMemcpyHostToDevice);

	Scan(devTestArr, MAX_NUMBER + 1);

	cudaMemcpy(testArr, devTestArr, sizeof(int) * (MAX_NUMBER + 1), cudaMemcpyDeviceToHost);

	for (int i = MAX_NUMBER + 1 - 16; i < MAX_NUMBER + 1; i++)
		fprintf(stderr, "%d ", testArr[i]);

	int* devTest;
	int hostTest = 0;
	cudaMalloc((void**)&devTest, sizeof(int));
	cudaMemcpy(devTest, &hostTest, sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemset(&devTest, 0, sizeof(int));
	TestAdd << <16384, 1024 >> > (devTest);
	
	cudaMemcpy(&hostTest, devTest, sizeof(int), cudaMemcpyDeviceToHost);
	fprintf(stderr, "TEST: %d\n", hostTest);*/

	int size;
	//cin >> size;
	fread(&size, sizeof(int), 1, stdin);

	auto hostArray = new int[size];
	fread(hostArray, sizeof(int), size, stdin);

	/*fprintf(stderr, "size = %d\n", size);
	for (int i = 0; i < size; i++)
	{
		//cin >> hostArray[i];
		//fprintf(stderr, "%d ", hostArray[i]);
	}*/

	int* devCount;
	CSC(cudaMalloc((void**)&devCount, sizeof(int) * (MAX_NUMBER + 1)));
	CSC(cudaMemset(devCount, 0, sizeof(int) * (MAX_NUMBER + 1)));

	int* devArray;
	CSC(cudaMalloc((void**)&devArray, sizeof(int) * size));
	CSC(cudaMemcpy(devArray, hostArray, sizeof(int) * size, cudaMemcpyHostToDevice));

	Histohram<<<256, 256>>>(devCount, devArray, size);
	cudaDeviceSynchronize();

	Scan(devCount, MAX_NUMBER + 1);

	int* outDevArray;
	CSC(cudaMalloc((void**)&outDevArray, sizeof(int) * size));
	CountSort<<<256, 256>>>(devCount, devArray, outDevArray, size);
	cudaDeviceSynchronize();

	CSC(cudaMemcpy(hostArray, outDevArray, sizeof(int) * size, cudaMemcpyDeviceToHost));

	fwrite(hostArray, sizeof(int), size, stdout);

	/*fprintf(stderr, "output:\n");

	for (int i = 0; i < size; i++)
	{
		//fprintf(stderr, "%d ", hostArray[i]);
	}*/
}