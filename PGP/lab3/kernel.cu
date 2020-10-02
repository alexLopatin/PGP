#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits>

#define CSC(call)                   \
do {                                \
    cudaError_t res = call;         \
    if (res != cudaSuccess) {       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                    \
    }                               \
} while(0)


#define Color uchar4

__constant__ double3 _classAvgs[32];

__global__ void Kernel(Color* out, unsigned int w, unsigned int h, int classCount)
{
	int startX = blockDim.x * blockIdx.x + threadIdx.x;
	int startY = blockDim.y * blockIdx.y + threadIdx.y;
	int stepX = blockDim.x * gridDim.x;
	int stepY = blockDim.y * gridDim.y;

	Color c;
	int i, j, k;

	for (i = startY; i < h; i += stepY)
	{
		for (j = startX; j < w; j += stepX)
		{
			c = out[i * w + j];

			double minVal = 255.0 * 255.0 * 255.0 * 255.0 * 255.0;
			int curClass = 0;

			for (k = 0; k < classCount; k++)
			{
				double newVal = (c.x - _classAvgs[k].x) * (c.x - _classAvgs[k].x)
					+ (c.y - _classAvgs[k].y) * (c.y - _classAvgs[k].y)
					+ (c.z - _classAvgs[k].z) * (c.z - _classAvgs[k].z);


				if (newVal < minVal)
				{
					minVal = newVal;
					curClass = k;
				}
			}

			c.w = curClass;
			out[i * w + j] = c;
		}
	}
}

__host__ void GetClassifiedImage(Color* result, unsigned int width, unsigned int height, int classCount)
{
	Color* deviceMap;
	CSC(cudaMalloc(&deviceMap, sizeof(Color) * width * height));
	CSC(cudaMemcpy(deviceMap, result, sizeof(Color) * width * height, cudaMemcpyHostToDevice));

	Kernel << <dim3(16, 16), dim3(16, 16) >> > (deviceMap, width, height, classCount);
	CSC(cudaDeviceSynchronize());

	CSC(cudaMemcpy(result, deviceMap, sizeof(Color) * width * height, cudaMemcpyDeviceToHost));
	CSC(cudaFree(deviceMap));
}

int main()
{
	std::string inFile;
	std::string outFile;
	int classCount;

	std::cin >> inFile >> outFile >> classCount;

	double3 hostAvgs[32];

	auto* file = fopen(inFile.c_str(), "rb");

	unsigned int width, height;
	fread(&width, sizeof(int), 1, file);
	fread(&height, sizeof(int), 1, file);

	auto hostMap = (Color*)malloc(sizeof(Color) * width * height);
	fread(hostMap, sizeof(Color), width * height, file);

	fclose(file);


	for (int i = 0; i < classCount; i++)
	{
		int pixelCount;
		std::cin >> pixelCount;
		double3 avg = {0,0,0};

		for (int j = 0; j < pixelCount; j++)
		{
			int x, y;
			std::cin >> x >> y;

			avg.x += hostMap[y * width + x].x;
			avg.y += hostMap[y * width + x].y;
			avg.z += hostMap[y * width + x].z;
		}

		avg.x = avg.x / pixelCount;
		avg.y = avg.y / pixelCount;
		avg.z = avg.z / pixelCount;

		hostAvgs[i] = avg;
	}

	CSC(cudaMemcpyToSymbol(_classAvgs, hostAvgs, sizeof(hostAvgs)));

	GetClassifiedImage(hostMap, width, height, classCount);

	file = fopen(outFile.c_str(), "wb");

	fwrite(&width, sizeof(int), 1, file);
	fwrite(&height, sizeof(int), 1, file);
	fwrite(hostMap, sizeof(Color), width * height, file);

	fclose(file);

	free(hostMap);
}
