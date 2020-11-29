#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include <thrust/extrema.h>
#include <thrust/device_vector.h>

using namespace thrust;

const double EPSILON = 10E-8;

struct Comparator
{
	__host__ __device__ bool operator()(double a, double b)
	{
		return fabs(a) < fabs(b) && fabs(b) >= 10E-8;
	}
};

__global__ void SwapRows(double* deviceMatrix,
	int currentRow,
	int otherRow,
	int currentColumn,
	int rowCount,
	int columnCount)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;

	for (auto i = idx; i < columnCount; i += offsetx)
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
	int idx = blockDim.x * blockIdx.x + threadIdx.x + currentColumn + 1;
	int offsetx = blockDim.x * gridDim.x;

	for (auto i = idx; i < columnCount; i += offsetx)
	{
		deviceMatrix[i * rowCount + currentRow] /= deviceMatrix[currentColumn * rowCount + currentRow];
	}
}

__global__ void CalculateRows(double* deviceMatrix,
	int currentRow,
	int currentColumn,
	int rowCount,
	int columnCount)
{
	int idx = threadIdx.x + currentRow + 1;
	int offsetx = blockDim.x;
	int idy = blockIdx.x + currentColumn + 1;
	int offsety = gridDim.x;

	for (int j = idx; j < rowCount; j += offsetx)
	{
		for (int k = idy; k < columnCount; k += offsety)
		{
			deviceMatrix[k * rowCount + j] -=
				deviceMatrix[currentColumn * rowCount + j] * deviceMatrix[k * rowCount + currentRow];
		}
	}
}

__global__ void SetCurrentZero(double* deviceMatrix,
	int currentRow,
	int currentColumn,
	int rowCount,
	int columnCount)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;

	for (auto i = idx; i < columnCount; i += offsetx)
	{
		deviceMatrix[i * rowCount + currentRow] = 0;
		deviceMatrix[currentColumn * rowCount + i] = 0;
	}
}

Comparator comparator;

__host__ int GetMaxIndexInColumn(double* deviceMatrix,
	int rowIndex,
	int columnIndex,
	int rowCount,
	int columnCount)
{
	if (columnIndex * rowCount == 0)
	{
		auto indexPointer = device_pointer_cast(deviceMatrix + columnIndex * rowCount);
		auto maxIndexPointer = max_element(indexPointer + rowIndex, indexPointer + rowCount, comparator);
		auto maxIndex = maxIndexPointer - indexPointer;

		auto elem = 0.0;
		cudaMemcpy(&elem, deviceMatrix + columnIndex * rowCount + maxIndex, sizeof(double), cudaMemcpyDeviceToHost);

		if (fabs(elem) < EPSILON)
		{
			return -1;
		}

		return maxIndex;
	}
	else
	{
		auto indexPointer = device_pointer_cast(deviceMatrix + columnIndex * rowCount);
		auto maxIndexPointer = thrust::max_element(indexPointer - 1 + rowIndex, indexPointer + rowCount, comparator);
		auto maxIndex = maxIndexPointer - indexPointer;
		if (maxIndex == rowIndex - 1)
		{
			return -1;
		}

		return maxIndex;
	}
}

__host__ int FindRank(double* matrix, int rowCount, int columnCount, int size)
{
	double* deviceMatrix;
	cudaMalloc(&deviceMatrix, rowCount * columnCount * sizeof(double));
	cudaMemcpy(deviceMatrix, matrix, rowCount * columnCount * sizeof(double), cudaMemcpyHostToDevice);

	auto offset = 0;

	for (int i = 0; i < rowCount && i + offset < columnCount; i++)
	{
		auto maxIndex = GetMaxIndexInColumn(deviceMatrix, i, i + offset, rowCount, columnCount);

		if (maxIndex < 0)
		{
			offset++;
			i--;
			continue;
		}

		if (maxIndex != i)
		{
			SwapRows<<<size, size >>>(deviceMatrix, i, maxIndex, i + offset, rowCount, columnCount);
		}

		CalculateCurrentRow<<<size, size >>>(deviceMatrix, i, i + offset, rowCount, columnCount);
		CalculateRows<<<size, size >>>(deviceMatrix, i, i + offset, rowCount, columnCount);
		SetCurrentZero<<<size, size >>>(deviceMatrix, i, i + offset, rowCount, columnCount);
	}

	cudaFree(deviceMatrix);

	auto rank = columnCount - offset > rowCount
		? rowCount
		: columnCount - offset;

	return rank;
}

int main()
{
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(nullptr);

	int rowCount, columnCount;
	std::cin >> rowCount >> columnCount;

	auto isTransposed = rowCount < columnCount;

	if (isTransposed)
	{
		auto temp = rowCount;
		rowCount = columnCount;
		columnCount = temp;
	}

	auto matrix = new double[rowCount * columnCount];

	for (int i = 0; i < rowCount; i++)
	{
		for (int j = 0; j < columnCount; j++)
		{
			//std::cin >> (isTransposed
			//	? matrix[i * columnCount + j]
			//	: matrix[j * rowCount + i]);
			matrix[i * columnCount + j] = rand() % 200 - 100;
		}
	}

	cudaEvent_t start, stop;
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);

		auto rank = FindRank(matrix, rowCount, columnCount, 32);
		//std:: cout << rank << '\n';

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << milliseconds / 1000 << ' ';
	}

	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);

		auto rank = FindRank(matrix, rowCount, columnCount, 1024);
		//std:: cout << rank << '\n';

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << milliseconds / 1000 << std::endl;
	}

	delete[] matrix;
}
