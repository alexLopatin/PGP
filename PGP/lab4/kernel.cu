#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include <thrust/extrema.h>
#include <thrust/device_vector.h>

using namespace thrust;

const double EPSILON = 10E-7;

double Abs(double number)
{
	return number > 0
		? number
		: -number;
}

struct Comparator
{
	__host__ __device__ bool operator()(double a, double b)
	{
		return fabs(a) < fabs(b);
	}
};

__global__ void SwapRows(double* deviceMatrix, int currentRow, int otherRow, int rowCount, int columnCount)
{
	auto start = blockDim.x * blockIdx.x + threadIdx.x;
	auto step = blockDim.x * gridDim.x;

	for (auto i = start; i < columnCount; i += step)
	{
		auto temp = deviceMatrix[i * rowCount + currentRow];
		deviceMatrix[i * rowCount + currentRow] = deviceMatrix[i * rowCount + otherRow];
		deviceMatrix[i * rowCount + otherRow] = temp;
	}
}

__global__ void CalculateCurrentRow(double* deviceMatrix,
	int currentRow,
	int currentColumn,
	int rowCount,
	int columnCount)
{
	auto start = blockDim.x * blockIdx.x + threadIdx.x + currentColumn + 1;
	auto step = blockDim.x * gridDim.x;

	for (auto i = start; i < columnCount; i += step)
	{
		deviceMatrix[i * rowCount + currentRow] /= deviceMatrix[currentColumn * rowCount + currentRow];
	}
}

__global__ void CalculateCurrentColumn(double* deviceMatrix,
	int currentRow,
	int currentColumn,
	int rowCount,
	int columnCount)
{
	auto start = blockDim.x * blockIdx.x + threadIdx.x + currentRow;
	auto step = blockDim.x * gridDim.x;

	for (auto i = start; i < rowCount; i += step)
	{
		deviceMatrix[currentColumn * rowCount + i] = (i == currentRow);
		printf("%f\n", deviceMatrix[currentColumn * rowCount + i]);
	}
}

__global__ void CalculateRows(double* deviceMatrix,
	int currentRow,
	int currentColumn,
	int rowCount,
	int columnCount)
{
	auto startX = blockDim.x * blockIdx.x + threadIdx.x + currentColumn + 1;
	auto startY = blockDim.y * blockIdx.y + threadIdx.y + currentRow + 1;
	auto stepX = blockDim.x * gridDim.x;
	auto stepY = blockDim.y * gridDim.y;

	for (auto i = startY; i < rowCount; i += stepY)
	{
		for (auto j = startX; j < columnCount; j += stepX)
		{
			deviceMatrix[j * rowCount + i] -= deviceMatrix[currentColumn * rowCount + i] * deviceMatrix[j * rowCount + currentRow];
		}
	}
}

__host__ int GetMaxIndexInColumn(double* deviceMatrix,
	int rowCount,
	int columnCount,
	int rowIndex,
	int columnIndex)
{
	Comparator comparator;

	auto indexPointer = device_pointer_cast(deviceMatrix + rowIndex * rowCount);
	auto maxIndexPointer = thrust::max_element(indexPointer + columnIndex, indexPointer + rowCount, comparator);
	auto maxIndex = maxIndexPointer - indexPointer;

	return maxIndex;
}

__host__ void PrintMatrix(double* matrix, int rowCount, int columnCount)
{
	for (int i = 0; i < rowCount; i++)
	{
		for (int j = 0; j < columnCount; j++)
		{
			std::cerr << matrix[j * rowCount + i] << ' ';
		}

		std::cerr << '\n';
	}
}

__host__ int FindRank(double* matrix, int rowCount, int columnCount)
{
	double* deviceMatrix;
	cudaMalloc(&deviceMatrix, rowCount * columnCount * sizeof(double));
	cudaMemcpy(deviceMatrix, matrix, rowCount * columnCount * sizeof(double), cudaMemcpyHostToDevice);

	auto offset = 0;

	try
	{
		for (int i = 0; i < rowCount - 1 && i + offset < columnCount; i++)
		{
			auto maxIndex = GetMaxIndexInColumn(deviceMatrix, rowCount, columnCount, i + offset, i);

			std::cerr << '\n';
			std::cerr << "pick max from:\n";
			for (int j = (i + offset) * rowCount + i; j < (i + offset) * rowCount + rowCount; j++)
			{
				std::cerr << matrix[j] << ' ';
			}
			std::cerr << '\n';

			std::cerr << "max elem: " << matrix[(i + offset) * rowCount + maxIndex] << '\n';

			if (Abs(matrix[(i + offset) * rowCount + maxIndex]) < EPSILON)
			{
				offset++;
				i--;
				continue;
			}

			if (maxIndex != i)
			{
				SwapRows << <1024, 1024 >> > (deviceMatrix, i, maxIndex, rowCount, columnCount);
				cudaThreadSynchronize();
			}


			CalculateCurrentRow << <1024, 1024 >> > (deviceMatrix, i, i + offset, rowCount, columnCount);
			cudaThreadSynchronize();

			CalculateRows << <dim3(32, 32), dim3(32, 32) >> > (deviceMatrix, i, i + offset, rowCount, columnCount);
			cudaThreadSynchronize();

			CalculateCurrentColumn << <1024, 1024 >> >(deviceMatrix, i, i + offset, rowCount, columnCount);
			cudaThreadSynchronize();

			//cudaMemcpy(matrix, deviceMatrix, rowCount * columnCount * sizeof(double), cudaMemcpyDeviceToHost);

			//PrintMatrix(matrix, rowCount, columnCount);
		}
	}
	catch (std::runtime_error& e)
	{
		std::cerr << rowCount << ' ' << columnCount << '\n';
		std::cerr << "offset: " << offset << '\n';
		std::cerr << e.what() << '\n';
	}

	cudaMemcpy(matrix, deviceMatrix, rowCount * columnCount * sizeof(double), cudaMemcpyDeviceToHost);
	PrintMatrix(matrix, rowCount, columnCount);

	auto rank = 0;

	for (int i = 0; i < rowCount; i++)
	{
		auto isZero = true;

		for (int j = 0; j < columnCount; j++)
		{
			if (Abs(matrix[j * rowCount + i]) > EPSILON)
			{
				isZero = false;
				break;
			}
		}

		rank += !isZero;
	}

	cudaFree(deviceMatrix);

	return rank;
}

int main()
{
	int rowCount, columnCount;
	std::cin >> rowCount >> columnCount;

	auto matrix = new double[rowCount * columnCount];

	for (int i = 0; i < rowCount; i++)
	{
		for (int j = 0; j < columnCount; j++)
		{
			std::cin >> matrix[j * rowCount + i];
		}
	}

	auto rank = FindRank(matrix, rowCount, columnCount);
	if (rank < 0)
		return -1;
	std::cout << rank << std::endl;

	PrintMatrix(matrix, rowCount, columnCount);

	delete[] matrix;
}
