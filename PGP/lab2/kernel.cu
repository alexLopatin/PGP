#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define Color uchar4

cudaArray* _arrayMap;
texture<Color, 2, cudaReadModeElementType> _textureMap;

__host__ void CreateImage(Color* hostMap, int width, int height)
{
	auto description = cudaCreateChannelDesc<Color>();

	cudaMallocArray(&_arrayMap, &description, width, height);
	cudaMemcpyToArray(_arrayMap, 0, 0, hostMap, sizeof(Color) * width * height, cudaMemcpyHostToDevice);

	_textureMap.addressMode[0] = cudaAddressModeClamp;
	_textureMap.addressMode[1] = cudaAddressModeClamp;
	_textureMap.channelDesc = description;
	_textureMap.filterMode = cudaFilterModePoint;
	_textureMap.normalized = false;

	cudaBindTextureToArray(_textureMap, _arrayMap, description);
}

__host__ void FreeImage()
{
	cudaUnbindTexture(_textureMap);
	cudaFreeArray(_arrayMap);
}

__device__ float4 Sum(float4 a, Color b, float coef)
{
	return {a.x + (float)b.x * coef,
		a.y + (float)b.y * coef,
		a.z + (float)b.z * coef,
		a.w + (float)b.w * coef};
}

__device__ Color ToColor(float4 a, float normalizeValue)
{
	return {(unsigned char)(a.x / normalizeValue),
		(unsigned char)(a.y / normalizeValue),
		(unsigned char)(a.z / normalizeValue),
		(unsigned char)(a.w / normalizeValue)};
}

__global__ void Kernel(Color* out, int w, int h, int radius, bool isX)
{
	float PI = 3.14159265359;

	int startX = blockDim.x * blockIdx.x + threadIdx.x;
	int startY = blockDim.y * blockIdx.y + threadIdx.y;
	int stepX = blockDim.x * gridDim.x;
	int stepY = blockDim.y * gridDim.y;

	Color c;
	float4 newColor;

	float r = radius != 0
		? radius
		: 1;
	float sum = 0;
	float coef = 0;

	int i, j, k;

	for (i = startY; i < h; i += stepY)
	{
		for (j = startX; j < w; j += stepX)
		{
			newColor = {0,0,0,0};
			sum = 0;

			for (k = -radius; k <= radius; k++)
			{
				c = isX
					? tex2D(_textureMap, j + k, i)
					: tex2D(_textureMap, j, i + k);
				coef = exp(-(float)(k * k) / (2 * r * r)) / (r * sqrt(2 * PI));

				newColor = Sum(newColor,
					c,
					coef);
				sum += coef;
			}
			
			out[i * w + j] = ToColor(newColor, sum);
		}
	}
}

__host__ void GetFilteredImage(Color* result, int width, int height, int radius)
{
	Color* deviceMap;
	cudaMalloc(&deviceMap, sizeof(Color) * width * height);

	cudaEvent_t start, stop;
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);

		//Y
		Kernel << <dim3(1, 1), dim3(32, 1) >> > (deviceMap, width, height, radius, false);
		cudaDeviceSynchronize();

		cudaMemcpyToArray(_arrayMap, 0, 0, deviceMap, sizeof(Color) * width * height, cudaMemcpyDeviceToDevice);

		//X
		Kernel << <dim3(1, 1), dim3(32, 1) >> > (deviceMap, width, height, radius, true);
		cudaDeviceSynchronize();

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
	}

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	{
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	//Y
	Kernel << <dim3(32, 32), dim3(32, 32) >> > (deviceMap, width, height, radius, false);
	cudaDeviceSynchronize();

	cudaMemcpyToArray(_arrayMap, 0, 0, deviceMap, sizeof(Color) * width * height, cudaMemcpyDeviceToDevice);

	//X
	Kernel << <dim3(32, 32), dim3(32, 32) >> > (deviceMap, width, height, radius, true);
	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	}

	float millisecondsSecond = 0;
	cudaEventElapsedTime(&millisecondsSecond, start, stop);

	std::cout << milliseconds / 1000 << ' ' << millisecondsSecond/1000 << std::endl;

	cudaMemcpy(result, deviceMap, sizeof(Color) * width * height, cudaMemcpyDeviceToHost);
	cudaFree(deviceMap);
}

int main()
{
	std::string inFile;
	std::string outFile;
	int radius;

	std::cin >> inFile >> outFile >> radius;

	auto *file = fopen(inFile.c_str(), "rb");

	int width, height;
	fread(&width, sizeof(int), 1, file);
	fread(&height, sizeof(int), 1, file);

	auto hostMap = (Color*)malloc(sizeof(Color) * width * height);
	fread(hostMap, sizeof(Color), width * height, file);

	fclose(file);

	CreateImage(hostMap, width, height);
	GetFilteredImage(hostMap, width, height, radius);

	file = fopen(outFile.c_str(), "wb");

	fwrite(&width, sizeof(int), 1, file);
	fwrite(&height, sizeof(int), 1, file);
	fwrite(hostMap, sizeof(Color), width * height, file);

	fclose(file);

	free(hostMap);
	FreeImage();
}
