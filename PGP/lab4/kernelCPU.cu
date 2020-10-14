#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <ctime>

using namespace thrust;

const double EPSILON = 10E-8;

void SwapRows(double* matrix,
	int currentRow,
	int otherRow,
	int currentColumn,
	int rowCount,
	int columnCount)
{
	for (auto i = 0; i < columnCount; i++)
	{
		auto temp = matrix[i * rowCount + currentRow];
		matrix[i * rowCount + currentRow] = matrix[i * rowCount + otherRow];
		matrix[i * rowCount + otherRow] = temp;
	}
}

void CalculateCurrentRow(double* matrix,
	int currentRow,
	int currentColumn,
	int rowCount,
	int columnCount)
{
	for (auto i = 0; i < columnCount; i++)
	{
		matrix[i * rowCount + currentRow] /= matrix[currentColumn * rowCount + currentRow];
	}
}

void CalculateRows(double* matrix,
	int currentRow,
	int currentColumn,
	int rowCount,
	int columnCount)
{
	for (int j = 0; j < rowCount; j++)
	{
		for (int k = 0; k < columnCount; k++)
		{
			matrix[k * rowCount + j] -=
				matrix[currentColumn * rowCount + j] * matrix[k * rowCount + currentRow];
		}
	}
}

void SetCurrentZero(double* matrix,
	int currentRow,
	int currentColumn,
	int rowCount,
	int columnCount)
{
	for (auto i = 0; i < columnCount; i++)
	{
		matrix[i * rowCount + currentRow] = 0;
		matrix[currentColumn * rowCount + i] = 0;
	}
}

int GetMaxIndexInColumn(double* deviceMatrix,
	int rowIndex,
	int columnIndex,
	int rowCount,
	int columnCount)
{
	auto maxElem = 0.0;
	auto maxIndex = 0;

	for (int i = columnIndex * rowCount + rowIndex; i < columnIndex * rowCount + rowCount; i++)
	{
		if (fabs(deviceMatrix[i]) > maxElem)
		{
			maxIndex = i;
			maxElem = fabs(deviceMatrix[i]);
		}
	}

	if (fabs(maxElem) < EPSILON)
	{
		return -1;
	}

	return maxIndex - columnIndex * rowCount;
}

int FindRank(double* matrix, int rowCount, int columnCount)
{
	auto offset = 0;

	for (int i = 0; i < rowCount && i + offset < columnCount; i++)
	{
		auto maxIndex = GetMaxIndexInColumn(matrix, i, i + offset, rowCount, columnCount);

		if (maxIndex < 0)
		{
			offset++;
			i--;
			continue;
		}

		if (maxIndex != i)
		{
			SwapRows(matrix, i, maxIndex, i + offset, rowCount, columnCount);
		}

		CalculateCurrentRow(matrix, i, i + offset, rowCount, columnCount);
		CalculateRows(matrix, i, i + offset, rowCount, columnCount);
		SetCurrentZero(matrix, i, i + offset, rowCount, columnCount);
	}

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

	clock_t begin = clock();

	for (int i = 0; i < rowCount; i++)
	{
		for (int j = 0; j < columnCount; j++)
		{
			std::cin >> (isTransposed
				? matrix[i * columnCount + j]
				: matrix[j * rowCount + i]);

		}
	}

	auto rank = FindRank(matrix, rowCount, columnCount);

	clock_t end = clock();

	std::cout << double(end - begin) / CLOCKS_PER_SEC << std::endl;

	delete[] matrix;
}
