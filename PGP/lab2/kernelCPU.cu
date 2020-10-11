#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <algorithm>

#define Color uchar4

float4 Sum(float4 a, Color b, float coef)
{
	return { a.x + (float)b.x * coef,
		a.y + (float)b.y * coef,
		a.z + (float)b.z * coef,
		a.w + (float)b.w * coef };
}

Color ToColor(float4 a, float normalizeValue)
{
	return { (unsigned char)(a.x / normalizeValue),
		(unsigned char)(a.y / normalizeValue),
		(unsigned char)(a.z / normalizeValue),
		(unsigned char)(a.w / normalizeValue) };
}

int GetCoordinate(int x, int y, int w, int h)
{
	int x_n = std::max(0, std::min(w - 1, x));
	int y_n = std::max(0, std::min(h - 1, y));

	return y_n * w + x_n;
}

void Kernel(Color* out, int w, int h, int radius, bool isX)
{
	float PI = 3.14159265359;

	Color c;
	float4 newColor;

	float r = radius != 0
		? radius
		: 1;
	float sum = 0;
	float coef = 0;

	int i, j, k;

	for (i = 0; i < h; i++)
	{
		for (j = 0; j < w; j++)
		{
			newColor = { 0,0,0,0 };
			sum = 0;

			for (k = -radius; k <= radius; k++)
			{
				int x = std::max(0, std::min(w - 1, j));
				int y = std::max(0, std::min(h - 1, i));

				c = isX
					? out[GetCoordinate(j + k, i, w, h)]
					: out[GetCoordinate(j, i + k, w, h)];
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

void GetFilteredImage(Color* result, int width, int height, int radius)
{
	//Y
	Kernel(result, width, height, radius, false);

	//X
	Kernel(result, width, height, radius, true);
}

int main()
{
	std::string inFile;
	std::string outFile;
	int radius;

	std::cin >> inFile >> outFile >> radius;
	radius /= 10000.0;
	auto* file = fopen(inFile.c_str(), "rb");

	int width, height;
	fread(&width, sizeof(int), 1, file);
	fread(&height, sizeof(int), 1, file);

	auto hostMap = (Color*)malloc(sizeof(Color) * width * height);
	fread(hostMap, sizeof(Color), width * height, file);

	fclose(file);

	clock_t begin = clock();
	GetFilteredImage(hostMap, width, height, radius);
	clock_t end = clock();

	std::cout << double(end - begin) / CLOCKS_PER_SEC << std::endl;

	file = fopen(outFile.c_str(), "wb");

	fwrite(&width, sizeof(int), 1, file);
	fwrite(&height, sizeof(int), 1, file);
	fwrite(hostMap, sizeof(Color), width * height, file);

	fclose(file);

	free(hostMap);
}
