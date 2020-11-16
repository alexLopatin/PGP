#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "mpi.h"

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

int main(int argc, char* argv[])
{
	int numproc, id;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	auto isMainProcess = !id;

	if (isMainProcess)
	{
		ReadInputData();
	}

	MPI_Bcast(&GridSize, 3, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&BlockSize, 3, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&SpaceSize, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&BoundaryConditions, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&InitValue, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	auto errorData = new double[numproc];
	auto data = (double*)malloc(sizeof(double) * (BlockSize[0] + 2) * (BlockSize[1] + 2) * (BlockSize[2] + 2));
	auto next = (double*)malloc(sizeof(double) * (BlockSize[0] + 2) * (BlockSize[1] + 2) * (BlockSize[2] + 2));
	auto temp = data;

	auto maxDim = *std::max_element(BlockSize, BlockSize + 3);
	auto buffIn = new double*[6];
	for(int i = 0; i < 6; i++)
		buffIn[i] = (double*)malloc(sizeof(double) * (maxDim + 2) * (maxDim + 2));
	auto buffOut = new double* [6];
	for (int i = 0; i < 6; i++)
		buffOut[i] = (double*)malloc(sizeof(double) * (maxDim + 2) * (maxDim + 2));

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
			for (int j = 0; j < BlockSize[1]; j++)
				for (int k = 0; k < BlockSize[2]; k++)
					buffOut[0][j + k * BlockSize[1]] = data[GetIndex(BlockSize[0] - 1, j, k)];
			MPI_Isend(buffOut[0], BlockSize[1] * BlockSize[2], MPI_DOUBLE, GetBlockIndex(bx + 1, by, bz), id, MPI_COMM_WORLD, &sendRequests[0]);
		}

		if (by < GridSize[1] - 1) {
			for (int i = 0; i < BlockSize[0]; i++)
				for (int k = 0; k < BlockSize[2]; k++)
					buffOut[1][i + k * BlockSize[0]] = data[GetIndex(i, BlockSize[1] - 1, k)];
			MPI_Isend(buffOut[1], BlockSize[0] * BlockSize[2], MPI_DOUBLE, GetBlockIndex(bx, by + 1, bz), id, MPI_COMM_WORLD, &sendRequests[1]);
		}

		if (bz < GridSize[2] - 1) {
			for (int i = 0; i < BlockSize[0]; i++)
				for (int j = 0; j < BlockSize[1]; j++)
					buffOut[2][i + j * BlockSize[0]] = data[GetIndex(i, j, BlockSize[2] - 1)];
			MPI_Isend(buffOut[2], BlockSize[0] * BlockSize[1], MPI_DOUBLE, GetBlockIndex(bx, by, bz + 1), id, MPI_COMM_WORLD, &sendRequests[2]);
		}

		if (bx > 0) {
			for (int j = 0; j < BlockSize[1]; j++)
				for (int k = 0; k < BlockSize[2]; k++)
					buffOut[3][j + k * BlockSize[1]] = data[GetIndex(0, j, k)];
			MPI_Isend(buffOut[3], BlockSize[1] * BlockSize[2], MPI_DOUBLE, GetBlockIndex(bx - 1, by, bz), id, MPI_COMM_WORLD, &sendRequests[3]);
		}

		if (by > 0) {
			for (int i = 0; i < BlockSize[0]; i++)
				for (int k = 0; k < BlockSize[2]; k++)
					buffOut[4][i + k * BlockSize[0]] = data[GetIndex(i, 0, k)];
			MPI_Isend(buffOut[4], BlockSize[0] * BlockSize[2], MPI_DOUBLE, GetBlockIndex(bx, by - 1, bz), id, MPI_COMM_WORLD, &sendRequests[4]);
		}

		if (bz > 0) {
			for (int i = 0; i < BlockSize[0]; i++)
				for (int j = 0; j < BlockSize[1]; j++)
					buffOut[5][i + j * BlockSize[0]] = data[GetIndex(i, j, 0)];
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

		if (bx > 0)
		{
			for (int j = 0; j < BlockSize[1]; j++)
				for (int k = 0; k < BlockSize[2]; k++)
					data[GetIndex(-1, j, k)] = buffIn[0][j + k * BlockSize[1]];
		}
		else
		{
			for (int j = 0; j < BlockSize[1]; j++)
				for (int k = 0; k < BlockSize[2]; k++)
					data[GetIndex(-1, j, k)] = BoundaryConditions[Side::Left];
		}

		if (by > 0)
		{
			for (int i = 0; i < BlockSize[0]; i++)
				for (int k = 0; k < BlockSize[2]; k++)
					data[GetIndex(i, -1, k)] = buffIn[1][i + k * BlockSize[0]];
		}
		else
		{
			for (int i = 0; i < BlockSize[0]; i++)
				for (int k = 0; k < BlockSize[2]; k++)
					data[GetIndex(i, -1, k)] = BoundaryConditions[Side::Front];
		}

		if (bz > 0)
		{
			for (int i = 0; i < BlockSize[0]; i++)
				for (int j = 0; j < BlockSize[1]; j++)
					data[GetIndex(i, j, -1)] = buffIn[2][i + j * BlockSize[0]];
		}
		else
		{
			for (int i = 0; i < BlockSize[0]; i++)
				for (int j = 0; j < BlockSize[1]; j++)
					data[GetIndex(i, j, -1)] = BoundaryConditions[Side::Down];
		}

		if (bx < GridSize[0] - 1)
		{
			for (int j = 0; j < BlockSize[1]; j++)
				for (int k = 0; k < BlockSize[2]; k++)
					data[GetIndex(BlockSize[0], j, k)] = buffIn[3][j + k * BlockSize[1]];
		}
		else
		{
			for (int j = 0; j < BlockSize[1]; j++)
				for (int k = 0; k < BlockSize[2]; k++)
					data[GetIndex(BlockSize[0], j, k)] = BoundaryConditions[Side::Right];
		}

		if (by < GridSize[1] - 1)
		{

			for (int i = 0; i < BlockSize[0]; i++)
				for (int k = 0; k < BlockSize[2]; k++)
					data[GetIndex(i, BlockSize[1], k)] = buffIn[4][i + k * BlockSize[0]];
		}
		else
		{
			for (int i = 0; i < BlockSize[0]; i++)
				for (int k = 0; k < BlockSize[2]; k++)
					data[GetIndex(i, BlockSize[1], k)] = BoundaryConditions[Side::Back];
		}

		if (bz < GridSize[2] - 1)
		{
			for (int i = 0; i < BlockSize[0]; i++)
				for (int j = 0; j < BlockSize[1]; j++)
					data[GetIndex(i, j, BlockSize[2])] = buffIn[5][i + j * BlockSize[0]];
		}
		else
		{
			for (int i = 0; i < BlockSize[0]; i++)
				for (int j = 0; j < BlockSize[1]; j++)
					data[GetIndex(i, j, BlockSize[2])] = BoundaryConditions[Side::Up];
		}

		//Перерасчет
		for (int i = 0; i < BlockSize[0]; i++)
			for (int j = 0; j < BlockSize[1]; j++)
				for (int k = 0; k < BlockSize[2]; k++)
					next[GetIndex(i, j, k)] =
					((data[GetIndex(i + 1, j, k)] + data[GetIndex(i - 1, j, k)]) / h2x +
						(data[GetIndex(i, j + 1, k)] + data[GetIndex(i, j - 1, k)]) / h2y +
						(data[GetIndex(i, j, k + 1)] + data[GetIndex(i, j, k - 1)]) / h2z)
					/ (2 * (1 / h2x + 1 / h2y + 1 / h2z));

		//Метрика
		error = 0.0;
		for (int i = 0; i < BlockSize[0]; i++)
			for (int j = 0; j < BlockSize[1]; j++)
				for (int k = 0; k < BlockSize[2]; k++)
					error = Max(error, Abs(next[GetIndex(i, j, k)] - data[GetIndex(i, j, k)]));

		MPI_Allgather(&error, 1, MPI_DOUBLE, errorData, 1, MPI_DOUBLE, MPI_COMM_WORLD);

		for (int i = 0; i < numproc; i++)
			error = Max(error, errorData[i]);

		temp = next;
		next = data;
		data = temp;
	}
	while (error >= Eps);

	MPI_Barrier(MPI_COMM_WORLD);

	if (!isMainProcess)
	{
		for (int k = 0; k < BlockSize[2]; k++)
		{
			for (int j = 0; j < BlockSize[1]; j++)
			{
				for (int i = -1; i <= BlockSize[0]; i++)
					buffIn[0][i + 1] = data[GetIndex(i, j, k)];
				MPI_Send(buffIn[0], (BlockSize[0] + 2), MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
			}
		}
	}
	else
	{
		std::ofstream output(FilePath, std::ios::out);
		output << std::scientific << std::setprecision(7);

		for (int kb = 0; kb < GridSize[2]; kb++)
		{
			for (int k = 0; k < BlockSize[2]; k++)
			{
				for (int jb = 0; jb < GridSize[1]; jb++)
				{
					for (int j = 0; j < BlockSize[1]; j++)
					{
						for (int ib = 0; ib < GridSize[0]; ib++)
						{
							if (GetBlockIndex(ib, jb, kb) == 0)
								for (int i = -1; i <= BlockSize[0]; i++)
									buffIn[0][i + 1] = data[GetIndex(i, j, k)];
							else
								MPI_Recv(buffIn[0], (BlockSize[0] + 2), MPI_DOUBLE, GetBlockIndex(ib, jb, kb), GetBlockIndex(ib, jb, kb), MPI_COMM_WORLD, &status);

							for (int i = 0; i < BlockSize[0]; i++)
							{
								//std::cerr << buff[0][i + 1 + (j + 1) * (BlockSize[0] + 2)] << ' ';
								output << buffIn[0][i + 1] << ' ';
							}
						}
					}
				}
			}
		}
	}

	MPI_Finalize();

	free(data);
	free(next);

	return 0;
}