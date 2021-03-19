#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h> 
#include <math.h>
#include <iostream>
#include <ctime>
//#include "mpi.h"
#include "omp.h"
#include "../../../../../../../Program Files (x86)/Microsoft SDKs/MPI/Include/mpi.h"

typedef unsigned char uchar;

struct vec3
{
	double x;
	double y;
	double z;
};

struct Triangle
{
	vec3 a;
	vec3 b;
	vec3 c;
	uchar4 color;
	// https://www.youtube.com/watch?v=PnVRnHBWhSk
	uchar4 ___;
	double reflection;
	double refraction;
};

struct Light
{
	vec3 pos;
	uchar4 color;
	// https://www.youtube.com/watch?v=PnVRnHBWhSk
	uchar4 ___;
};

struct Hit
{
	vec3 pos;
	vec3 normal;
	int trigId;
};

struct MotionParams
{
	double r0, Ar, Wr, Pr;
	double z0, Az, Wz, Pz;
	double fi0, Wfi;
};

int frames = 16;
char path[256] = "res/%d.data";
int w = 640;
int h = 480;
double degree = 120;

int maxRecur = 4;
int sqrRaySSAA = 2;

MotionParams cameraParams;
MotionParams cameraViewDirParams;

Light lights[4] =
{
	{{-6.0, -6.0, 8.0}, {0, 128, 0}},
	{{6.0, 2.0, 10.0}, {255, 0, 0}}
};
int lightCount = 2;

#define trigsCount 62
Triangle trigs[trigsCount];

__device__ __host__ double dot(vec3 a, vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ vec3 prod(vec3 a, vec3 b) {
	return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

__device__ __host__ vec3 norm(vec3 v) {
	double l = sqrt(dot(v, v));
	return { v.x / l, v.y / l, v.z / l };
}

__device__ __host__ vec3 diff(vec3 a, vec3 b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

__device__ __host__ vec3 add(vec3 a, vec3 b) {
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__device__ __host__ vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {
	return { a.x * v.x + b.x * v.y + c.x * v.z,
				a.y * v.x + b.y * v.y + c.y * v.z,
				a.z * v.x + b.z * v.y + c.z * v.z };
}

__device__ __host__ double Max(double a, double b)
{
	return a > b
		? a
		: b;
}

__device__ __host__ double Min(double a, double b)
{
	return a > b
		? b
		: a;
}

bool ContainsKey(char* arr[], int count, char* key)
{
	for (int i = 0; i < count; i++)
	{
		if (strcmp(arr[i], key) == 0)
		{
			return true;
		}
	}

	return false;
}

vec3 ToCartesian(vec3 cylindric)
{
	return
	{
		cylindric.x * cos(cylindric.y),
		cylindric.x * sin(cylindric.y),
		cylindric.z
	};
}

void BuildFloor(vec3 a, vec3 b, vec3 c, vec3 d, uchar4 color, double refl)
{
	trigs[0] = { a, c, b, color, {}, refl };
	trigs[1] = { a, d, c, color, {}, refl };
}

void BuildTetra(vec3 pos, uchar4 color, double r, double refl, double refr)
{
	double c = r / sqrt(3);

	trigs[2] = { {c, c, c}, {-c, c, -c}, {c, -c, -c}, color, {}, refl, refr };
	trigs[3] = { {c, c, c}, {-c, -c, c}, {-c, c, -c}, color, {}, refl, refr };
	trigs[4] = { {c, c, c}, {-c, -c, c}, {c, -c, -c}, color, {}, refl, refr };
	trigs[5] = { {-c, c, -c}, {-c, -c, c}, {c, -c, -c}, color, {}, refl, refr };

	for (int i = 2; i < 6; i++)
	{
		trigs[i].a.x += pos.x; trigs[i].b.x += pos.x; trigs[i].c.x += pos.x;
		trigs[i].a.y += pos.y; trigs[i].b.y += pos.y; trigs[i].c.y += pos.y;
		trigs[i].a.z += pos.z; trigs[i].b.z += pos.z; trigs[i].c.z += pos.z;
	}
}

void BuildIcosa(vec3 pos, uchar4 color, double r, double refl, double refr)
{
	double c = r / 1.118;

	vec3 vertices[12] =
	{
		{0.809, 0.5, 0.588},
		{0.309, -0.5, 0.951},
		{-0.309, 0.5, 0.951},
		{-0.809, -0.5, 0.588},
		{-1, 0.5, 0},
		{-0.809, -0.5, -0.588},
		{-0.309, 0.5, -0.951},
		{0.309, -0.5, -0.951},
		{0.809, 0.5, -0.588},
		{1, -0.5, 0},
		{0, 1.118, 0},
		{0, -1.118, 0}
	};

	trigs[6] = { vertices[0], vertices[10], vertices[2], color, {}, refl, refr };
	trigs[7] = { vertices[2], vertices[10], vertices[4], color, {}, refl, refr };
	trigs[8] = { vertices[4], vertices[10], vertices[6], color, {}, refl, refr };
	trigs[9] = { vertices[6], vertices[10], vertices[8], color, {}, refl, refr };
	trigs[10] = { vertices[8], vertices[10], vertices[0],color, {}, refl, refr };
	trigs[11] = { vertices[0], vertices[2], vertices[1], color, {}, refl, refr };
	trigs[12] = { vertices[1], vertices[2], vertices[3], color, {}, refl, refr };
	trigs[13] = { vertices[2], vertices[4], vertices[3], color, {}, refl, refr };
	trigs[14] = { vertices[3], vertices[4], vertices[5], color, {}, refl, refr };
	trigs[15] = { vertices[4], vertices[6], vertices[5], color, {}, refl, refr };
	trigs[16] = { vertices[5], vertices[6], vertices[7], color, {}, refl, refr };
	trigs[17] = { vertices[6], vertices[8], vertices[7], color, {}, refl, refr };
	trigs[18] = { vertices[7], vertices[8], vertices[9], color, {}, refl, refr };
	trigs[19] = { vertices[8], vertices[0], vertices[9], color, {}, refl, refr };
	trigs[20] = { vertices[9], vertices[0], vertices[1], color, {}, refl, refr };
	trigs[21] = { vertices[1], vertices[11], vertices[9],color, {}, refl, refr };
	trigs[22] = { vertices[3], vertices[11], vertices[1],color, {}, refl, refr };
	trigs[23] = { vertices[5], vertices[11], vertices[3],color, {}, refl, refr };
	trigs[24] = { vertices[7], vertices[11], vertices[5],color, {}, refl, refr };
	trigs[25] = { vertices[9], vertices[11], vertices[7],color, {}, refl, refr };

	for (int i = 6; i < 26; i++)
	{
		trigs[i].a.x *= c; trigs[i].b.x *= c; trigs[i].c.x *= c;
		trigs[i].a.y *= c; trigs[i].b.y *= c; trigs[i].c.y *= c;
		trigs[i].a.z *= c; trigs[i].b.z *= c; trigs[i].c.z *= c;

		trigs[i].a.x += pos.x; trigs[i].b.x += pos.x; trigs[i].c.x += pos.x;
		trigs[i].a.y += pos.y; trigs[i].b.y += pos.y; trigs[i].c.y += pos.y;
		trigs[i].a.z += pos.z; trigs[i].b.z += pos.z; trigs[i].c.z += pos.z;
	}
}

void BuildDodeca(vec3 pos, uchar4 color, double r, double refl, double refr)
{
	double c = r / 1.118;

	vec3 vertices[21] =
	{
		{},
		{ 0.577350, 0.577350, -0.57735 },
		{ 0.577350, -0.57735, -0.57735 },
		{ 0.577350, 0.577350, 0.577350 },
		{ 0.577350, -0.57735, 0.577350 },
		{ -0.57735, 0.577350, -0.57735 },
		{ -0.57735, -0.57735, -0.57735 },
		{ -0.57735, 0.577350, 0.577350 },
		{ -0.57735, -0.57735, 0.577350 },
		{ 0.356822, 0.000000, -0.93417 },
		{ -0.35682, 0.000000, -0.93417 },
		{ 0.356822, 0.000000, 0.934172 },
		{ -0.35682, 0.000000, 0.934172 },
		{ 0.934172, 0.356822, 0.000000 },
		{ 0.934172, -0.35682, 0.000000 },
		{ -0.93417, 0.356822, 0.000000 },
		{ -0.93417, -0.35682, 0.000000 },
		{ 0.000000, 0.934172, -0.35682 },
		{ 0.000000, 0.934172, 0.356822 },
		{ 0.000000, -0.93417, -0.35682 },
		{ 0.000000, -0.93417, 0.356822 }
	};

	trigs[26] = { vertices[9], vertices[17], vertices[1] ,  color, {}, refl, refr };
	trigs[27] = { vertices[9], vertices[10], vertices[5] ,  color, {}, refl, refr };
	trigs[28] = { vertices[13], vertices[9], vertices[1] ,  color, {}, refl, refr };
	trigs[29] = { vertices[13], vertices[14], vertices[2] , color, {}, refl, refr };
	trigs[30] = { vertices[17], vertices[13], vertices[1] , color, {}, refl, refr };
	trigs[31] = { vertices[17], vertices[18], vertices[3] , color, {}, refl, refr };
	trigs[32] = { vertices[2], vertices[10], vertices[9] ,  color, {}, refl, refr };
	trigs[33] = { vertices[2], vertices[19], vertices[6] ,  color, {}, refl, refr };
	trigs[34] = { vertices[3], vertices[14], vertices[13],  color, {}, refl, refr };
	trigs[35] = { vertices[3], vertices[11], vertices[4] ,  color, {}, refl, refr };
	trigs[36] = { vertices[5], vertices[18], vertices[17],  color, {}, refl, refr };
	trigs[37] = { vertices[5], vertices[15], vertices[7] ,  color, {}, refl, refr };
	trigs[38] = { vertices[6], vertices[5], vertices[10],   color, {}, refl, refr };
	trigs[39] = { vertices[6], vertices[16], vertices[15],  color, {}, refl, refr };
	trigs[40] = { vertices[12], vertices[18], vertices[7] , color, {}, refl, refr };
	trigs[41] = { vertices[12], vertices[11], vertices[3] , color, {}, refl, refr };
	trigs[42] = { vertices[20], vertices[14], vertices[4] , color, {}, refl, refr };
	trigs[43] = { vertices[20], vertices[19], vertices[2] , color, {}, refl, refr };
	trigs[44] = { vertices[16], vertices[20], vertices[8] , color, {}, refl, refr };
	trigs[45] = { vertices[16], vertices[6], vertices[19],  color, {}, refl, refr };
	trigs[46] = { vertices[12], vertices[16], vertices[8] , color, {}, refl, refr };
	trigs[47] = { vertices[12], vertices[7], vertices[15],  color, {}, refl, refr };
	trigs[48] = { vertices[20], vertices[12], vertices[8] , color, {}, refl, refr };
	trigs[49] = { vertices[20], vertices[4], vertices[11],  color, {}, refl, refr };
	trigs[50] = { vertices[9], vertices[5], vertices[17],   color, {}, refl, refr };
	trigs[51] = { vertices[13], vertices[2], vertices[9] ,  color, {}, refl, refr };
	trigs[52] = { vertices[17], vertices[3], vertices[13],  color, {}, refl, refr };
	trigs[53] = { vertices[2], vertices[6], vertices[10],   color, {}, refl, refr };
	trigs[54] = { vertices[3], vertices[4], vertices[14],   color, {}, refl, refr };
	trigs[55] = { vertices[5], vertices[7], vertices[18],   color, {}, refl, refr };
	trigs[56] = { vertices[6], vertices[15], vertices[5] ,  color, {}, refl, refr };
	trigs[57] = { vertices[12], vertices[3], vertices[18],  color, {}, refl, refr };
	trigs[58] = { vertices[20], vertices[2], vertices[14],  color, {}, refl, refr };
	trigs[59] = { vertices[16], vertices[19], vertices[20], color, {}, refl, refr };
	trigs[60] = { vertices[12], vertices[15], vertices[16], color, {}, refl, refr };
	trigs[61] = { vertices[20], vertices[11], vertices[12], color, {}, refl, refr };

	for (int i = 26; i < 62; i++)
	{
		trigs[i].a.x *= c; trigs[i].b.x *= c; trigs[i].c.x *= c;
		trigs[i].a.y *= c; trigs[i].b.y *= c; trigs[i].c.y *= c;
		trigs[i].a.z *= c; trigs[i].b.z *= c; trigs[i].c.z *= c;

		trigs[i].a.x += pos.x; trigs[i].b.x += pos.x; trigs[i].c.x += pos.x;
		trigs[i].a.y += pos.y; trigs[i].b.y += pos.y; trigs[i].c.y += pos.y;
		trigs[i].a.z += pos.z; trigs[i].b.z += pos.z; trigs[i].c.z += pos.z;
	}
}

__device__ __host__ Hit RayHit(vec3 pos, vec3 dir, Triangle* trigs)
{
	int k, k_min = -1;
	double ts_min = 0.0;
	vec3 hitPos = { 0, 0, 0 };
	vec3 normal = { 0, 0, 0 };
	for (k = 0; k < trigsCount; k++) {
		vec3 e1 = diff(trigs[k].b, trigs[k].a);
		vec3 e2 = diff(trigs[k].c, trigs[k].a);
		vec3 p = prod(dir, e2);
		double div = dot(p, e1);
		if (fabs(div) < 1e-10)
			continue;
		vec3 t = diff(pos, trigs[k].a);
		double u = dot(p, t) / div;
		if (u < 0.0 || u > 1.0)
			continue;
		vec3 q = prod(t, e1);
		double v = dot(q, dir) / div;
		if (v < 0.0 || v + u > 1.0)
			continue;
		double ts = dot(q, e2) / div;
		if (ts < 0.0)
			continue;

		if (k_min == -1 || ts < ts_min)
		{
			k_min = k;
			ts_min = ts;
			hitPos = add({dir.x * ts, dir.y * ts, dir.z * ts}, pos);
			normal = norm(prod(e1, e2));
		}
	}

	return { hitPos, normal, k_min };
}

__device__ __host__ uchar4 Ray(
	vec3 pos, vec3 dir, vec3 cameraDir,
	Triangle* trigs, Light* lights, int lightsCount,
	int maxRecur, int curRecur = 1)
{
	Hit rayHit = RayHit(pos, dir, trigs);

	if (rayHit.trigId == -1)
	{
		return { 0, 0, 0, 1 };
	}

	vec3 rayHitPos =
	{
		rayHit.pos.x + 0.001,
		rayHit.pos.y + 0.001,
		rayHit.pos.z + 0.001
	};

	double r = trigs[rayHit.trigId].reflection;
	double dotA = dot(dir, rayHit.normal);
	vec3 a =
	{
		rayHit.normal.x * 2 * dotA,
		rayHit.normal.y * 2 * dotA,
		rayHit.normal.z * 2 * dotA
	};
	vec3 reflDir = norm(diff(dir, a));

	uchar4 res = {0, 0, 0, 0};

	for (int i = 0; i < lightsCount; i++)
	{
		Light light = lights[i];
		vec3 lightDir = norm(diff(light.pos, rayHitPos));
		Hit lightHit = RayHit(rayHitPos, lightDir, trigs);
		// ambient
		double4 lightColor = { 0.3, 0.3, 0.3, 0 };

		if (lightHit.trigId == -1)
		{
			double diffuse = dot(rayHit.normal, lightDir);
			double specular = powf(dot(dir, reflDir), 32) + 1.0;

			lightColor =
			{
				Min(light.color.x * diffuse / 255.0 * specular + lightColor.x, 1),
				Min(light.color.y * diffuse / 255.0 * specular + lightColor.y, 1),
				Min(light.color.z * diffuse / 255.0 * specular + lightColor.z, 1),
				(double)light.color.w
			};;
		}

		res =
		{
			(uchar)Min(trigs[rayHit.trigId].color.x * lightColor.x + res.x, 255),
			(uchar)Min(trigs[rayHit.trigId].color.y * lightColor.y + res.y, 255),
			(uchar)Min(trigs[rayHit.trigId].color.z * lightColor.z + res.z, 255),
			(uchar)(res.w + 1)
		};
	}

	if (r > 0.01 && curRecur < maxRecur)
	{
		uchar4 refl = Ray(rayHitPos, reflDir, cameraDir, trigs, lights, lightsCount, maxRecur, ++curRecur);

		res =
		{
			(uchar)(res.x * (1 - r) + r * refl.x),
			(uchar)(res.y * (1 - r) + r * refl.y),
			(uchar)(res.z * (1 - r) + r * refl.z),
			(uchar)(res.w + refl.w)
		};
	}

	double refr = trigs[rayHit.trigId].refraction;

	if (refr < 0.99 && curRecur < maxRecur)
	{
		rayHitPos =
		{
			rayHit.pos.x + 0.01 * dir.x,
			rayHit.pos.y + 0.01 * dir.y,
			rayHit.pos.z + 0.01 * dir.z
		};
		uchar4 refracted = Ray(rayHitPos, dir, cameraDir, trigs, lights, lightsCount, maxRecur, ++curRecur);

		res =
		{
			(uchar)(res.x * (1 - refr) + refr * refracted.x),
			(uchar)(res.y * (1 - refr) + refr * refracted.y),
			(uchar)(res.z * (1 - refr) + refr * refracted.z),
			(uchar)(res.w + refracted.w)
		};
	}

	return res;
}

__global__ void Render(
	vec3 pc, vec3 pv,
	int w, int h, double angle,
	uchar4* data, Triangle* trigs, Light* lights,
	int lightsCount, int maxRecur, int sqrRaySSAA)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsety = blockDim.y * gridDim.y;

	double dw = 2.0 / (w - 1.0);
	double dh = 2.0 / (h - 1.0);
	double z = 1.0 / tan(angle * M_PI / 360.0);
	vec3 bz = norm(diff(pv, pc));
	vec3 bx = norm(prod(bz, { 0.0, 0.0, 1.0 }));
	vec3 by = norm(prod(bx, bz));

	for (int i = idx; i < w; i += offsetx)
		for (int j = idy; j < h; j += offsety) {
			vec3 c = { -1.0, -1.0, z };
			vec3 cameraDir = norm(mult(bx, by, bz, c));

			//SSAA
			if (sqrRaySSAA > 1)
			{
				double4 res = { 0, 0, 0, 0 };
				double r = sqrRaySSAA;

				for (int k = 0; k < r; k++)
				{
					for (int p = 0; p < r; p++)
					{
						vec3 v = { -1.0 + dw * (i + k / r - r / 2), (-1.0 + dh * (j + p / r - r / 2)) * h / w, z };
						vec3 dir = mult(bx, by, bz, v);

						uchar4 aaRay = Ray(pc, norm(dir), cameraDir, trigs, lights, lightsCount, maxRecur);
						res =
						{
							res.x + aaRay.x / (r * r),
							res.y + aaRay.y / (r * r),
							res.z + aaRay.z / (r * r),
							res.w + aaRay.w
						};
					}
				}

				data[(h - 1 - j) * w + i] =
				{
					(uchar)res.x,
					(uchar)res.y,
					(uchar)res.z,
					(uchar)res.w
				};
			}
			else
			{
				vec3 v = { -1.0 + dw * i, (-1.0 + dh * j) * h / w, z };
				vec3 dir = mult(bx, by, bz, v);

				data[(h - 1 - j) * w + i] = Ray(pc, norm(dir), cameraDir, trigs, lights, lightsCount, maxRecur);
			}
		}
}

__host__ void RenderCPU(
	vec3 pc, vec3 pv,
	int w, int h, double angle,
	uchar4* data, Triangle* trigs, Light* lights,
	int lightsCount, int maxRecur, int sqrRaySSAA)
{
	double dw = 2.0 / (w - 1.0);
	double dh = 2.0 / (h - 1.0);
	double z = 1.0 / tan(angle * M_PI / 360.0);
	vec3 bz = norm(diff(pv, pc));
	vec3 bx = norm(prod(bz, { 0.0, 0.0, 1.0 }));
	vec3 by = norm(prod(bx, bz));

	#pragma omp parallel
	{
		#pragma omp for
		for (int l = 0; l < w * h; l++) {
			int i = l / h;
			int j = l % w;
			vec3 c = { -1.0, -1.0, z };
			vec3 cameraDir = norm(mult(bx, by, bz, c));

			//SSAA
			if (sqrRaySSAA > 1)
			{
				double4 res = { 0, 0, 0, 0 };
				double r = sqrRaySSAA;

				for (int k = 0; k < r; k++)
				{
					for (int p = 0; p < r; p++)
					{
						vec3 v = { -1.0 + dw * (i + k / r - r / 2), (-1.0 + dh * (j + p / r - r / 2)) * h / w, z };
						vec3 dir = mult(bx, by, bz, v);

						uchar4 aaRay = Ray(pc, norm(dir), cameraDir, trigs, lights, lightsCount, maxRecur);
						res =
						{
							res.x + aaRay.x / (r * r),
							res.y + aaRay.y / (r * r),
							res.z + aaRay.z / (r * r),
							res.w + aaRay.w
						};
					}
				}

				data[(h - 1 - j) * w + i] = { (uchar)res.x,
					(uchar)res.y,
					(uchar)res.z,
					(uchar)res.w };
			}
			else
			{
				vec3 v = { -1.0 + dw * i, (-1.0 + dh * j) * h / w, z };
				vec3 dir = mult(bx, by, bz, v);

				data[(h - 1 - j) * w + i] = Ray(pc, norm(dir), cameraDir, trigs, lights, lightsCount, maxRecur);
			}
		}
	}
}

void ReadParams()
{
	std::cin >> frames;
	std::cin >> path;
	std::cin >> w >> h;
	std::cin >> degree;

	std::cin >> cameraParams.r0 >> cameraParams.z0 >> cameraParams.fi0
		>> cameraParams.Ar >> cameraParams.Az
		>> cameraParams.Wr >> cameraParams.Wz >> cameraParams.Wfi
		>> cameraParams.Pr >> cameraParams.Pz;

	std::cin >> cameraViewDirParams.r0 >> cameraViewDirParams.z0 >> cameraViewDirParams.fi0
		>> cameraViewDirParams.Ar >> cameraViewDirParams.Az
		>> cameraViewDirParams.Wr >> cameraViewDirParams.Wz >> cameraViewDirParams.Wfi
		>> cameraViewDirParams.Pr >> cameraViewDirParams.Pz;

	vec3 pos;
	double3 color;
	double radius, reflection, refraction, lightEdgeCount = 0;

	std::cin >> pos.x >> pos.y >> pos.z
		>> color.x >> color.y >> color.z
		>> radius >> reflection >> refraction >> lightEdgeCount;
	BuildTetra(pos, { (uchar)(color.x * 255), (uchar)(color.y * 255), (uchar)(color.z * 255), 0 }, radius, reflection, refraction);

	std::cin >> pos.x >> pos.y >> pos.z
		>> color.x >> color.y >> color.z
		>> radius >> reflection >> refraction >> lightEdgeCount;
	BuildIcosa(pos, { (uchar)(color.x * 255), (uchar)(color.y * 255), (uchar)(color.z * 255), 0 }, radius, reflection, refraction);

	std::cin >> pos.x >> pos.y >> pos.z
		>> color.x >> color.y >> color.z
		>> radius >> reflection >> refraction >> lightEdgeCount;
	BuildDodeca(pos, { (uchar)(color.x * 255), (uchar)(color.y * 255), (uchar)(color.z * 255), 0 }, radius, reflection, refraction);

	vec3 floor[4];
	char texturePath[256];
	for (int i = 0; i < 4; i++)
	{
		std::cin >> floor[i].x >> floor[i].y >> floor[i].z;
	}
	std::cin >> texturePath;
	std::cin >> color.x >> color.y >> color.z;
	std::cin >> reflection;
	BuildFloor(floor[0], floor[1], floor[2], floor[3], { (uchar)(color.x * 255), (uchar)(color.y * 255), (uchar)(color.z * 255), 0 }, reflection);

	std::cin >> lightCount;
	for (int i = 0; i < lightCount; i++)
	{
		std::cin >> pos.x >> pos.y >> pos.z
			>> color.x >> color.y >> color.z;
		lights[i] = { pos, { (uchar)(color.x * 255), (uchar)(color.y * 255), (uchar)(color.z * 255), 0 } };
	}

	std::cin >> maxRecur >> sqrRaySSAA;

	//maxRecur <= 4
	maxRecur = Min(maxRecur, 4);
}

void SetDefaultParams()
{
	cameraParams =
	{
		8, 2, 1, 0,
		8, 1, 1, 0,
		0, 1
	};
	cameraViewDirParams =
	{
		0, 0, 0, 0,
		0, 0, 0, 0,
		2, 0
	};

	BuildFloor({ -100, -100, 0 }, { -100, 100, 0 }, { 100, 100, 0 }, { 100, -100, 0 }, { 64, 64, 64, 0 }, 0.5);
	BuildTetra({ 0, 0, 3 }, { 128, 0, 0, 0 }, 4, 0, 0.5);
	BuildIcosa({ -4, -4, 5 }, { 0, 128, 0, 0 }, 2, 0, 0);
	BuildDodeca({ 6, -6, 3 }, { 128, 128, 0, 0 }, 2, 0, 0);
}

int main(int argc, char* argv[])
{
	int numproc, id, numDevice;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	cudaGetDeviceCount(&numDevice);
	cudaSetDevice(id % numDevice);

	bool useCpu = ContainsKey(argv, argc, "--cpu");

	if (!id)
	{
		bool useDefault = ContainsKey(argv, argc, "--default");
		useDefault
			? SetDefaultParams()
			: ReadParams();
	}

	//send params to all processes
	MPI_Bcast(&useCpu, 1, MPI_BYTE, 0, MPI_COMM_WORLD);

	MPI_Bcast(&frames, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&path, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
	MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&h, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&degree, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Bcast(&cameraParams, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&cameraViewDirParams, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Bcast(&trigs, trigsCount * 14, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Bcast(&lightCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lights, lightCount * 4, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Bcast(&maxRecur, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sqrRaySSAA, 1, MPI_INT, 0, MPI_COMM_WORLD);

	char buff[256];
	uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	vec3 pc, pv;

	Triangle* devTrigs;
	cudaMalloc(&devTrigs, sizeof(Triangle) * trigsCount);
	cudaMemcpy(devTrigs, trigs, sizeof(Triangle) * trigsCount, cudaMemcpyHostToDevice);

	Light* devLights;
	cudaMalloc(&devLights, sizeof(Light) * lightCount);
	cudaMemcpy(devLights, lights, sizeof(Light) * lightCount, cudaMemcpyHostToDevice);

	uchar4* devData;
	cudaMalloc(&devData, sizeof(uchar4) * w * h);

	float elapsed = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int first = frames / numproc * id;
	int last = (id != numDevice - 1)
		? frames / numproc * id
		: frames;

	for (int k = first; k < last; k++)
	{
		double t = k * 2 * M_PI / frames;
		pc = ToCartesian
		({
			cameraParams.r0 + cameraParams.Ar * sin(cameraParams.Wr * t + cameraParams.Pr),
			cameraParams.fi0 + cameraParams.Wfi * t,
			cameraParams.z0 + cameraParams.Az * sin(cameraParams.Wz * t + cameraParams.Pz)
		});
		pv = ToCartesian
		({
			cameraViewDirParams.r0 + cameraViewDirParams.Ar * sin(cameraViewDirParams.Wr * t + cameraViewDirParams.Pr),
			cameraViewDirParams.fi0 + cameraViewDirParams.Wfi * t,
			cameraViewDirParams.z0 + cameraViewDirParams.Az * sin(cameraViewDirParams.Wz * t + cameraViewDirParams.Pz)
		});

		if (!useCpu)
		{
			cudaEventRecord(start, 0);

			Render << <dim3(32, 32), dim3(16, 16) >> > (pc, pv, w, h, degree, devData, devTrigs, devLights, lightCount, maxRecur, sqrRaySSAA);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsed, start, stop);

			cudaMemcpy(data, devData, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost);
		}
		else
		{
			clock_t begin = clock();

			RenderCPU(pc, pv, w, h, degree, data, trigs, lights, lightCount, maxRecur, sqrRaySSAA);

			clock_t end = clock();
			elapsed = (float)(end - begin) / CLOCKS_PER_SEC * 1000;
		}

		int rayCount = 0;

		for (int i = 0; i < w * h; i++)
		{
			rayCount += data[i].w;
		}

		sprintf(buff, "res/%d.data", k);
		printf("%d\t%d\t%d\n", k, (int)elapsed, rayCount);

		FILE* out = fopen(buff, "wb");
		fwrite(&w, sizeof(int), 1, out);
		fwrite(&h, sizeof(int), 1, out);
		fwrite(data, sizeof(uchar4), w * h, out);
		fclose(out);
	}

	free(data);
	MPI_Finalize();

	return 0;
}
