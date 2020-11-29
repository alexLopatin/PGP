#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "mpi.h"

#include <thrust/extrema.h>
#include <thrust/device_vector.h>
//#include "../../../../../../../Program Files (x86)/Microsoft SDKs/MPI/Include/mpi.h"

using namespace thrust;

int GridSize[3];
int BlockSize[3];
double SpaceSize[3];

std::string FilePath;

double Eps;

double BoundaryConditions[6];
double InitValue;

enum Side
{
	Down, Up,
	Left, Right,
	Front, Back
};

struct Comparator
{
	__host__ __device__ bool operator()(double a, double b)
	{
		return a < b;
	}
};

Comparator comparator;

void ReadInputData()
{
	std::cin >> GridSize[0] >> GridSize[1] >> GridSize[2];
	std::cin >> BlockSize[0] >> BlockSize[1] >> BlockSize[2];

	std::cin >> FilePath;

	std::cin >> Eps;

	std::cin >> SpaceSize[0] >> SpaceSize[1] >> SpaceSize[2];
	std::cin
		>> BoundaryConditions[0] >> BoundaryConditions[1]
		>> BoundaryConditions[2] >> BoundaryConditions[3]
		>> BoundaryConditions[4] >> BoundaryConditions[5];
	std::cin >> InitValue;

	std::cerr << GridSize[0] << ' ' << GridSize[1] << ' ' << GridSize[2] << ' ';
	std::cerr << BlockSize[0] << ' ' << BlockSize[1] << ' ' << BlockSize[2] << ' ';
	std::cerr << BoundaryConditions[0] << ' ' << BoundaryConditions[1] << ' ' << BoundaryConditions[2] << ' '
		<< BoundaryConditions[3] << ' ' << BoundaryConditions[4] << ' ' << BoundaryConditions[5] << '\n';
	std::cerr << InitValue << std::endl;
	std::cerr << Eps << std::endl;
}

int GetIndex(int x, int y, int z)
{
	return (x + 1) + (y + 1) * (BlockSize[0] + 2) + (z + 1) * (BlockSize[0] + 2) * (BlockSize[1] + 2);
}

__device__ int DeviceGetIndex(int x, int y, int z, int blockSizeX, int blockSizeY, int blockSizeZ)
{
	return (x + 1) + (y + 1) * (blockSizeX + 2) + (z + 1) * (blockSizeX + 2) * (blockSizeY + 2);
}

int GetBlockIndex(int x, int y, int z)
{
	return x + y * GridSize[0] + z * GridSize[0] * GridSize[1];
}

double Max(double a, double b)
{
	return a > b
		? a
		: b;
}

double Abs(double a)
{
	return a > 0
		? a
		: -a;
}

__global__ void Kernel(
	double* data, double* next,
	int blockSizeX, int blockSizeY, int blockSizeZ,
	double h2x, double h2y, double h2z)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsety = blockDim.y * gridDim.y;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;
	int offsetz = blockDim.z * gridDim.z;

	for (int i = idx; i < blockSizeX; i += offsetx)
		for (int j = idy; j < blockSizeY; j += offsety)
			for (int k = idz; k < blockSizeZ; k += offsetz)
				next[DeviceGetIndex(i, j, k, blockSizeX, blockSizeY, blockSizeZ)] =
				((data[DeviceGetIndex(i + 1, j, k, blockSizeX, blockSizeY, blockSizeZ)] + data[DeviceGetIndex(i - 1, j, k, blockSizeX, blockSizeY, blockSizeZ)]) / h2x +
					(data[DeviceGetIndex(i, j + 1, k, blockSizeX, blockSizeY, blockSizeZ)] + data[DeviceGetIndex(i, j - 1, k, blockSizeX, blockSizeY, blockSizeZ)]) / h2y +
					(data[DeviceGetIndex(i, j, k + 1, blockSizeX, blockSizeY, blockSizeZ)] + data[DeviceGetIndex(i, j, k - 1, blockSizeX, blockSizeY, blockSizeZ)]) / h2z)
				/ (2 * (1 / h2x + 1 / h2y + 1 / h2z));
}

__global__ void CopyFace(
	double* data, double* dest,
	int blockSizeX, int blockSizeY, int blockSizeZ,
	Side side)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsety = blockDim.y * gridDim.y;

	switch (side)
	{
	case Side::Left:
		for (int j = idx; j < blockSizeY; j += offsetx)
			for (int k = idy; k < blockSizeZ; k += offsety)
				dest[j + k * blockSizeY] = data[DeviceGetIndex(0, j, k, blockSizeX, blockSizeY, blockSizeZ)];
		break;
	case Side::Front:
		for (int i = idx; i < blockSizeX; i += offsetx)
			for (int k = idy; k < blockSizeZ; k += offsety)
				dest[i + k * blockSizeX] = data[DeviceGetIndex(i, 0, k, blockSizeX, blockSizeY, blockSizeZ)];
		break;
	case Side::Down:
		for (int i = idx; i < blockSizeX; i += offsetx)
			for (int j = idy; j < blockSizeY; j += offsety)
				dest[i + j * blockSizeX] = data[DeviceGetIndex(i, j, 0, blockSizeX, blockSizeY, blockSizeZ)];
		break;
	case Side::Right:
		for (int j = idx; j < blockSizeY; j += offsetx)
			for (int k = idy; k < blockSizeZ; k += offsety)
				dest[j + k * blockSizeY] = data[DeviceGetIndex(blockSizeX - 1, j, k, blockSizeX, blockSizeY, blockSizeZ)];
		break;
	case Side::Back:
		for (int i = idx; i < blockSizeX; i += offsetx)
			for (int k = idy; k < blockSizeZ; k += offsety)
				dest[i + k * blockSizeX] = data[DeviceGetIndex(i, blockSizeY - 1, k, blockSizeX, blockSizeY, blockSizeZ)];
		break;
	case Side::Up:
		for (int i = idx; i < blockSizeX; i += offsetx)
			for (int j = idy; j < blockSizeY; j += offsety)
				dest[i + j * blockSizeX] = data[DeviceGetIndex(i, j, blockSizeZ - 1, blockSizeX, blockSizeY, blockSizeZ)];
		break;
	}
}

__global__ void PasteFace(
	double* data, double* source,
	int blockSizeX, int blockSizeY, int blockSizeZ,
	Side side)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsety = blockDim.y * gridDim.y;

	switch (side)
	{
	case Side::Left:
		for (int j = idx; j < blockSizeY; j += offsetx)
			for (int k = idy; k < blockSizeZ; k += offsety)
				data[DeviceGetIndex(-1, j, k, blockSizeX, blockSizeY, blockSizeZ)] = source[j + k * blockSizeY];
		break;
	case Side::Front:
		for (int i = idx; i < blockSizeX; i += offsetx)
			for (int k = idy; k < blockSizeZ; k += offsety)
				data[DeviceGetIndex(i, -1, k, blockSizeX, blockSizeY, blockSizeZ)] = source[i + k * blockSizeX];
		break;
	case Side::Down:
		for (int i = idx; i < blockSizeX; i += offsetx)
			for (int j = idy; j < blockSizeY; j += offsety)
				data[DeviceGetIndex(i, j, -1, blockSizeX, blockSizeY, blockSizeZ)] = source[i + j * blockSizeX];
		break;
	case Side::Right:
		for (int j = idx; j < blockSizeY; j += offsetx)
			for (int k = idy; k < blockSizeZ; k += offsety)
				data[DeviceGetIndex(blockSizeX, j, k, blockSizeX, blockSizeY, blockSizeZ)] = source[j + k * blockSizeY];
		break;
	case Side::Back:
		for (int i = idx; i < blockSizeX; i += offsetx)
			for (int k = idy; k < blockSizeZ; k += offsety)
				data[DeviceGetIndex(i, blockSizeY, k, blockSizeX, blockSizeY, blockSizeZ)] = source[i + k * blockSizeX];
		break;
	case Side::Up:
		for (int i = idx; i < blockSizeX; i += offsetx)
			for (int j = idy; j < blockSizeY; j += offsety)
				data[DeviceGetIndex(i, j, blockSizeZ, blockSizeX, blockSizeY, blockSizeZ)] = source[i + j * blockSizeX];
		break;
	}
}

__global__ void ErrorKernel(
	double* data, double* next,
	int blockSizeX, int blockSizeY, int blockSizeZ)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsety = blockDim.y * gridDim.y;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;
	int offsetz = blockDim.z * gridDim.z;

	for (int i = idx - 1; i < blockSizeX + 1; i += offsetx)
		for (int j = idy - 1; j < blockSizeY + 1; j += offsety)
			for (int k = idz - 1; k < blockSizeZ + 1; k += offsetz)
			{
				data[DeviceGetIndex(i, j, k, blockSizeX, blockSizeY, blockSizeZ)] = fabs(next[DeviceGetIndex(i, j, k, blockSizeX, blockSizeY, blockSizeZ)] - data[DeviceGetIndex(i, j, k, blockSizeX, blockSizeY, blockSizeZ)]);
				//printf("%f\n", data[DeviceGetIndex(i, j, k, blockSizeX, blockSizeY, blockSizeZ)]);
			}
}

int main(int argc, char* argv[])
{
	int numproc, id, numDevice;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	cudaGetDeviceCount(&numDevice);
	cudaSetDevice(id % numDevice);

	auto isMainProcess = !id;
	int filePathSize;

	if (isMainProcess)
	{
		ReadInputData();
		filePathSize = FilePath.size();
	}

	MPI_Bcast(&GridSize, 3, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&BlockSize, 3, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&SpaceSize, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&BoundaryConditions, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&InitValue, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


	MPI_Bcast(&filePathSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	FilePath.resize(filePathSize);
	MPI_Bcast((char*)FilePath.c_str(), filePathSize, MPI_CHAR, 0, MPI_COMM_WORLD);

	auto errorData = new double[numproc];
	auto data = (double*)malloc(sizeof(double) * (BlockSize[0] + 2) * (BlockSize[1] + 2) * (BlockSize[2] + 2));

	auto maxDim = *std::max_element(BlockSize, BlockSize + 3);
	auto buffIn = new double* [6];
	for (int i = 0; i < 6; i++)
		buffIn[i] = (double*)malloc(sizeof(double) * (maxDim + 2) * (maxDim + 2));
	auto buffOut = new double* [6];
	for (int i = 0; i < 6; i++)
		buffOut[i] = (double*)malloc(sizeof(double) * (maxDim + 2) * (maxDim + 2));

	memset(data, 0, sizeof(double) * (BlockSize[0] + 2) * (BlockSize[1] + 2) * (BlockSize[2] + 2));

	for (int i = 0; i < BlockSize[0]; i++)
	{
		for (int j = 0; j < BlockSize[1]; j++)
		{
			for (int k = 0; k < BlockSize[2]; k++)
			{
				data[GetIndex(i, j, k)] = InitValue;
			}
		}
	}

	double* deviceData;
	double* deviceNext;
	double* deviceBuff;

	cudaMalloc(&deviceData, sizeof(double) * (BlockSize[0] + 2) * (BlockSize[1] + 2) * (BlockSize[2] + 2));
	cudaMalloc(&deviceNext, sizeof(double) * (BlockSize[0] + 2) * (BlockSize[1] + 2) * (BlockSize[2] + 2));
	cudaMalloc(&deviceBuff, sizeof(double) * (maxDim + 2) * (maxDim + 2));
	cudaMemcpy(deviceData, data, sizeof(double) * (BlockSize[0] + 2) * (BlockSize[1] + 2) * (BlockSize[2] + 2), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceNext, data, sizeof(double) * (BlockSize[0] + 2) * (BlockSize[1] + 2) * (BlockSize[2] + 2), cudaMemcpyHostToDevice);

	auto temp = deviceData;

	auto error = 0.0;

	auto h2x = SpaceSize[0] / (GridSize[0] * BlockSize[0]) * SpaceSize[0] / (GridSize[0] * BlockSize[0]);
	auto h2y = SpaceSize[1] / (GridSize[1] * BlockSize[1]) * SpaceSize[1] / (GridSize[1] * BlockSize[1]);
	auto h2z = SpaceSize[2] / (GridSize[2] * BlockSize[2]) * SpaceSize[2] / (GridSize[2] * BlockSize[2]);

	auto bx = id % GridSize[0];
	auto by = (id / GridSize[0]) % GridSize[1];
	auto bz = (id / GridSize[0]) / GridSize[1];

	MPI_Request sendRequests[6], receiveRequests[6];
	for (int i = 0; i < 6; i++)
	{
		sendRequests[i] = MPI_REQUEST_NULL;
		receiveRequests[i] = MPI_REQUEST_NULL;
	}

	MPI_Status statuses[6];

	do
	{
		MPI_Barrier(MPI_COMM_WORLD);

		//Отправка

		if (bx < GridSize[0] - 1) {
			CopyFace << <32, 32 >> > (deviceData, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Right);
			cudaDeviceSynchronize();
			cudaMemcpy(buffOut[0], deviceBuff, sizeof(double) * BlockSize[1] * BlockSize[2], cudaMemcpyDeviceToHost);

			MPI_Isend(buffOut[0], BlockSize[1] * BlockSize[2], MPI_DOUBLE, GetBlockIndex(bx + 1, by, bz), id, MPI_COMM_WORLD, &sendRequests[0]);
		}

		if (by < GridSize[1] - 1) {
			CopyFace << <32, 32 >> > (deviceData, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Back);
			cudaDeviceSynchronize();
			cudaMemcpy(buffOut[1], deviceBuff, sizeof(double) * BlockSize[0] * BlockSize[2], cudaMemcpyDeviceToHost);

			MPI_Isend(buffOut[1], BlockSize[0] * BlockSize[2], MPI_DOUBLE, GetBlockIndex(bx, by + 1, bz), id, MPI_COMM_WORLD, &sendRequests[1]);
		}

		if (bz < GridSize[2] - 1) {
			CopyFace << <32, 32 >> > (deviceData, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Up);
			cudaDeviceSynchronize();
			cudaMemcpy(buffOut[2], deviceBuff, sizeof(double) * BlockSize[0] * BlockSize[1], cudaMemcpyDeviceToHost);

			MPI_Isend(buffOut[2], BlockSize[0] * BlockSize[1], MPI_DOUBLE, GetBlockIndex(bx, by, bz + 1), id, MPI_COMM_WORLD, &sendRequests[2]);
		}

		if (bx > 0) {
			CopyFace << <32, 32 >> > (deviceData, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Left);
			cudaDeviceSynchronize();
			cudaMemcpy(buffOut[3], deviceBuff, sizeof(double) * BlockSize[1] * BlockSize[2], cudaMemcpyDeviceToHost);

			MPI_Isend(buffOut[3], BlockSize[1] * BlockSize[2], MPI_DOUBLE, GetBlockIndex(bx - 1, by, bz), id, MPI_COMM_WORLD, &sendRequests[3]);
		}

		if (by > 0) {
			CopyFace << <32, 32 >> > (deviceData, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Front);
			cudaDeviceSynchronize();
			cudaMemcpy(buffOut[4], deviceBuff, sizeof(double) * BlockSize[0] * BlockSize[2], cudaMemcpyDeviceToHost);

			MPI_Isend(buffOut[4], BlockSize[0] * BlockSize[2], MPI_DOUBLE, GetBlockIndex(bx, by - 1, bz), id, MPI_COMM_WORLD, &sendRequests[4]);
		}

		if (bz > 0) {
			CopyFace << <32, 32 >> > (deviceData, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Down);
			cudaDeviceSynchronize();
			cudaMemcpy(buffOut[5], deviceBuff, sizeof(double) * BlockSize[0] * BlockSize[1], cudaMemcpyDeviceToHost);

			MPI_Isend(buffOut[5], BlockSize[0] * BlockSize[1], MPI_DOUBLE, GetBlockIndex(bx, by, bz - 1), id, MPI_COMM_WORLD, &sendRequests[5]);
		}

		//Получение

		if (bx > 0)
			MPI_Irecv(buffIn[0], BlockSize[1] * BlockSize[2], MPI_DOUBLE, GetBlockIndex(bx - 1, by, bz), GetBlockIndex(bx - 1, by, bz), MPI_COMM_WORLD, &receiveRequests[0]);
		if (by > 0)
			MPI_Irecv(buffIn[1], BlockSize[0] * BlockSize[2], MPI_DOUBLE, GetBlockIndex(bx, by - 1, bz), GetBlockIndex(bx, by - 1, bz), MPI_COMM_WORLD, &receiveRequests[1]);
		if (bz > 0)
			MPI_Irecv(buffIn[2], BlockSize[0] * BlockSize[1], MPI_DOUBLE, GetBlockIndex(bx, by, bz - 1), GetBlockIndex(bx, by, bz - 1), MPI_COMM_WORLD, &receiveRequests[2]);
		if (bx < GridSize[0] - 1)
			MPI_Irecv(buffIn[3], BlockSize[1] * BlockSize[2], MPI_DOUBLE, GetBlockIndex(bx + 1, by, bz), GetBlockIndex(bx + 1, by, bz), MPI_COMM_WORLD, &receiveRequests[3]);
		if (by < GridSize[1] - 1)
			MPI_Irecv(buffIn[4], BlockSize[0] * BlockSize[2], MPI_DOUBLE, GetBlockIndex(bx, by + 1, bz), GetBlockIndex(bx, by + 1, bz), MPI_COMM_WORLD, &receiveRequests[4]);
		if (bz < GridSize[2] - 1)
			MPI_Irecv(buffIn[5], BlockSize[0] * BlockSize[1], MPI_DOUBLE, GetBlockIndex(bx, by, bz + 1), GetBlockIndex(bx, by, bz + 1), MPI_COMM_WORLD, &receiveRequests[5]);

		MPI_Waitall(6, sendRequests, statuses);
		MPI_Waitall(6, receiveRequests, statuses);

		if (!(bx > 0))
		{
			for (int j = 0; j < BlockSize[1]; j++)
				for (int k = 0; k < BlockSize[2]; k++)
					buffIn[0][j + k * BlockSize[1]] = BoundaryConditions[Side::Left];
		}
		cudaMemcpy(deviceBuff, buffIn[0], sizeof(double) * BlockSize[1] * BlockSize[2], cudaMemcpyHostToDevice);
		PasteFace << <32, 32 >> > (deviceData, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Left);
		cudaDeviceSynchronize();
		PasteFace << <32, 32 >> > (deviceNext, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Left);
		cudaDeviceSynchronize();

		if (!(by > 0))
		{
			for (int i = 0; i < BlockSize[0]; i++)
				for (int k = 0; k < BlockSize[2]; k++)
					buffIn[1][i + k * BlockSize[0]] = BoundaryConditions[Side::Front];
		}
		cudaMemcpy(deviceBuff, buffIn[1], sizeof(double) * BlockSize[0] * BlockSize[2], cudaMemcpyHostToDevice);
		PasteFace << <32, 32 >> > (deviceData, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Front);
		cudaDeviceSynchronize();
		PasteFace << <32, 32 >> > (deviceNext, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Front);
		cudaDeviceSynchronize();

		if (!(bz > 0))
		{
			for (int i = 0; i < BlockSize[0]; i++)
				for (int j = 0; j < BlockSize[1]; j++)
					buffIn[2][i + j * BlockSize[0]] = BoundaryConditions[Side::Down];
		}
		cudaMemcpy(deviceBuff, buffIn[2], sizeof(double) * BlockSize[0] * BlockSize[1], cudaMemcpyHostToDevice);
		PasteFace << <32, 32 >> > (deviceData, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Down);
		cudaDeviceSynchronize();
		PasteFace << <32, 32 >> > (deviceNext, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Down);
		cudaDeviceSynchronize();

		if (!(bx < GridSize[0] - 1))
		{
			for (int j = 0; j < BlockSize[1]; j++)
				for (int k = 0; k < BlockSize[2]; k++)
					buffIn[3][j + k * BlockSize[1]] = BoundaryConditions[Side::Right];
		}
		cudaMemcpy(deviceBuff, buffIn[3], sizeof(double) * BlockSize[1] * BlockSize[2], cudaMemcpyHostToDevice);
		PasteFace << <32, 32 >> > (deviceData, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Right);
		cudaDeviceSynchronize();
		PasteFace << <32, 32 >> > (deviceNext, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Right);
		cudaDeviceSynchronize();

		if (!(by < GridSize[1] - 1))
		{
			for (int i = 0; i < BlockSize[0]; i++)
				for (int k = 0; k < BlockSize[2]; k++)
					buffIn[4][i + k * BlockSize[0]] = BoundaryConditions[Side::Back];
		}
		cudaMemcpy(deviceBuff, buffIn[4], sizeof(double) * BlockSize[0] * BlockSize[2], cudaMemcpyHostToDevice);
		PasteFace << <32, 32 >> > (deviceData, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Back);
		cudaDeviceSynchronize();
		PasteFace << <32, 32 >> > (deviceNext, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Back);
		cudaDeviceSynchronize();

		if (!(bz < GridSize[2] - 1))
		{
			for (int i = 0; i < BlockSize[0]; i++)
				for (int j = 0; j < BlockSize[1]; j++)
					buffIn[5][i + j * BlockSize[0]] = BoundaryConditions[Side::Up];
		}
		cudaMemcpy(deviceBuff, buffIn[5], sizeof(double) * BlockSize[0] * BlockSize[1], cudaMemcpyHostToDevice);
		PasteFace << <32, 32 >> > (deviceData, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Up);
		cudaDeviceSynchronize();
		PasteFace << <32, 32 >> > (deviceNext, deviceBuff, BlockSize[0], BlockSize[1], BlockSize[2], Side::Up);
		cudaDeviceSynchronize();

		Kernel << <16, 16, 16 >> > (
			deviceData, deviceNext,
			BlockSize[0], BlockSize[1], BlockSize[2],
			h2x, h2y, h2z);
		cudaDeviceSynchronize();
		ErrorKernel << <16, 16, 16 >> > (
			deviceData, deviceNext,
			BlockSize[0], BlockSize[1], BlockSize[2]);
		cudaDeviceSynchronize();

		auto indexPointer = device_pointer_cast(deviceData);
		error = *(max_element(indexPointer, indexPointer + (BlockSize[0] + 2) * (BlockSize[1] + 2) * (BlockSize[2] + 2), comparator));

		MPI_Allgather(&error, 1, MPI_DOUBLE, errorData, 1, MPI_DOUBLE, MPI_COMM_WORLD);

		for (int i = 0; i < numproc; i++)
			error = Max(error, errorData[i]);

		temp = deviceNext;
		deviceNext = deviceData;
		deviceData = temp;
	} while (error >= Eps);

	cudaMemcpy(data, deviceData, sizeof(double) * (BlockSize[0] + 2) * (BlockSize[1] + 2) * (BlockSize[2] + 2), cudaMemcpyDeviceToHost);

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_File fp;
	MPI_Datatype numChar, filetype, xySlice;

	auto doubleSize = 15;
	auto charData = new char[BlockSize[0] * BlockSize[1] * BlockSize[2] * doubleSize];

	for (int k = 0; k < BlockSize[2]; k++)
		for (int j = 0; j < BlockSize[1]; j++)
			for (int i = 0; i < BlockSize[0]; i++)
			{
				auto index = i + j * BlockSize[0] + k * BlockSize[0] * BlockSize[1];

				auto length = sprintf(&charData[index * doubleSize], "%.*e ", 7, data[GetIndex(i, j, k)]);

				if (length < doubleSize)
				{
					charData[index * doubleSize + length] = ' ';
				}
			}

	MPI_Type_contiguous(doubleSize, MPI_CHAR, &numChar);
	MPI_Type_commit(&numChar);

	MPI_Type_vector(BlockSize[1], BlockSize[0], BlockSize[0] * GridSize[0], numChar, &xySlice);
	MPI_Type_commit(&xySlice);

	MPI_Type_create_hvector(BlockSize[2], 1, doubleSize * GridSize[0] * GridSize[1] * BlockSize[0] * BlockSize[1], xySlice, &filetype);
	MPI_Type_commit(&filetype);

	auto offset = BlockSize[0] * bx
		+ BlockSize[1] * BlockSize[0] * GridSize[0] * by
		+ BlockSize[2] * BlockSize[1] * BlockSize[0] * GridSize[1] * GridSize[0] * bz;

	//MPI_File_delete(FilePath.c_str(), MPI_INFO_NULL);
	MPI_File_open(MPI_COMM_WORLD, FilePath.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
	MPI_File_set_view(fp, offset * doubleSize, MPI_CHAR, filetype, "native", MPI_INFO_NULL);

	MPI_File_write_all(fp, charData, BlockSize[0] * BlockSize[1] * BlockSize[2] * doubleSize, MPI_CHAR, MPI_STATUS_IGNORE);

	MPI_File_close(&fp);

	MPI_Finalize();

	return 0;
}