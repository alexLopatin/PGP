#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <ctime>

void ReadArray(double* arr, int size);
void WriteArray(double* arr, int size);

void Operate(double* firstDevice, double* secondDevice, double* resultDevice, int vectorLength);

double Min(double a, double b);

int main()
{
	int vectorLength;
	std::cin >> vectorLength;

	clock_t begin = clock();

	double* first;
	double* second;
	double* result;

	first = (double*)malloc(sizeof(double) * vectorLength);
	second = (double*)malloc(sizeof(double) * vectorLength);
	result = (double*)malloc(sizeof(double) * vectorLength);

	ReadArray(first, vectorLength);
	ReadArray(second, vectorLength);

	Operate(first, second, result, vectorLength);

	clock_t end = clock();

	std::cout << double(end - begin) / CLOCKS_PER_SEC << std::endl;

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

int Min(int a, int b)
{
	return (a > b)
		? b
		: a;
}

int Max(int a, int b)
{
	return (a < b)
		? b
		: a;
}

void Operate(double* firstDevice, double* secondDevice, double* resultDevice, int vectorLength)
{
	for (int i = 0; i < vectorLength - 1; i++)
	{
		resultDevice[i] = Min(firstDevice[i], secondDevice[i]);
	}
}

double Min(double a, double b)
{
	return (a > b)
		? b
		: a;
}