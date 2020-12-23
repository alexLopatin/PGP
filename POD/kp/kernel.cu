#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h> 
#include <math.h>

typedef unsigned char uchar;

/*struct uchar4 {
	uchar x;
	uchar y;
	uchar z;
	uchar w;
};*/

struct vec3 {
	double x;
	double y;
	double z;
};

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

struct trig {
	vec3 a;
	vec3 b;
	vec3 c;
	uchar4 color;
	double reflection;
	double refraction;
};

struct light {
	vec3 pos;
	uchar4 color;
};

struct hit {
	vec3 pos;
	vec3 normal;
	int trigId;
};

__device__ double Max(double a, double b)
{
	return a > b
		? a
		: b;
}

__device__ double Min(double a, double b)
{
	return a > b
		? b
		: a;
}

#define trigsCount 62

trig trigs[trigsCount];

#define sqrtPixelRayCount 1

void build_floor(vec3 a, vec3 b, vec3 c, vec3 d, uchar4 color, double refl)
{
	trigs[0] = { a, c, b, color, refl };
	trigs[1] = { a, d, c, color, refl };
}

void build_tetra(vec3 pos, uchar4 color, double r)
{
	double c = r / sqrt(3);

	trigs[2] = { {c, c, c}, {-c, c, -c}, {c, -c, -c}, color, 0.5, 0.5 };
	trigs[3] = { {c, c, c}, {-c, -c, c}, {-c, c, -c}, color, 0.5, 0.5 };
	trigs[4] = { {c, c, c}, {-c, -c, c}, {c, -c, -c}, color, 0.5, 0.5 };
	trigs[5] = { {-c, c, -c}, {-c, -c, c}, {c, -c, -c}, color, 0.5, 0.5 };

	for (int i = 2; i < 6; i++)
	{
		trigs[i].a.x += pos.x; trigs[i].b.x += pos.x; trigs[i].c.x += pos.x;
		trigs[i].a.y += pos.y; trigs[i].b.y += pos.y; trigs[i].c.y += pos.y;
		trigs[i].a.z += pos.z; trigs[i].b.z += pos.z; trigs[i].c.z += pos.z;
	}
}

void build_icosa(vec3 pos, uchar4 color, double r)
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

	trigs[6] = { vertices[0], vertices[10], vertices[2], color, 0.0 };
	trigs[7] = { vertices[2], vertices[10], vertices[4], color, 0.0 };
	trigs[8] = { vertices[4], vertices[10], vertices[6], color, 0.0 };
	trigs[9] = { vertices[6], vertices[10], vertices[8], color, 0.0 };
	trigs[10] = { vertices[8], vertices[10], vertices[0], color, 0.0 };
	trigs[11] = { vertices[0], vertices[2], vertices[1], color, 0.0 };
	trigs[12] = { vertices[1], vertices[2], vertices[3], color, 0.0 };
	trigs[13] = { vertices[2], vertices[4], vertices[3], color, 0.0 };
	trigs[14] = { vertices[3], vertices[4], vertices[5], color, 0.0 };
	trigs[15] = { vertices[4], vertices[6], vertices[5], color, 0.0 };
	trigs[16] = { vertices[5], vertices[6], vertices[7], color, 0.0 };
	trigs[17] = { vertices[6], vertices[8], vertices[7], color, 0.0 };
	trigs[18] = { vertices[7], vertices[8], vertices[9], color, 0.0 };
	trigs[19] = { vertices[8], vertices[0], vertices[9], color, 0.0 };
	trigs[20] = { vertices[9], vertices[0], vertices[1], color, 0.0 };
	trigs[21] = { vertices[1], vertices[11], vertices[9], color, 0.0 };
	trigs[22] = { vertices[3], vertices[11], vertices[1], color, 0.0 };
	trigs[23] = { vertices[5], vertices[11], vertices[3], color, 0.0 };
	trigs[24] = { vertices[7], vertices[11], vertices[5], color, 0.0 };
	trigs[25] = { vertices[9], vertices[11], vertices[7], color, 0.0 };

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

void build_dodeca(vec3 pos, uchar4 color, double r)
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

	trigs[26] = { vertices[9], vertices[17], vertices[1] , color, 0.0 };
	trigs[27] = { vertices[9], vertices[10], vertices[5] , color, 0.0 };
	trigs[28] = { vertices[13], vertices[9], vertices[1] , color, 0.0 };
	trigs[29] = { vertices[13], vertices[14], vertices[2] , color, 0.0 };
	trigs[30] = { vertices[17], vertices[13], vertices[1] , color, 0.0 };
	trigs[31] = { vertices[17], vertices[18], vertices[3] , color, 0.0 };
	trigs[32] = { vertices[2], vertices[10], vertices[9] , color, 0.0 };
	trigs[33] = { vertices[2], vertices[19], vertices[6] , color, 0.0 };
	trigs[34] = { vertices[3], vertices[14], vertices[13], color, 0.0 };
	trigs[35] = { vertices[3], vertices[11], vertices[4] , color, 0.0 };
	trigs[36] = { vertices[5], vertices[18], vertices[17], color, 0.0 };
	trigs[37] = { vertices[5], vertices[15], vertices[7] , color, 0.0 };
	trigs[38] = { vertices[6], vertices[5], vertices[10], color, 0.0 };
	trigs[39] = { vertices[6], vertices[16], vertices[15], color, 0.0 };
	trigs[40] = { vertices[12], vertices[18], vertices[7] , color, 0.0 };
	trigs[41] = { vertices[12], vertices[11], vertices[3] , color, 0.0 };
	trigs[42] = { vertices[20], vertices[14], vertices[4] , color, 0.0 };
	trigs[43] = { vertices[20], vertices[19], vertices[2] , color, 0.0 };
	trigs[44] = { vertices[16], vertices[20], vertices[8] , color, 0.0 };
	trigs[45] = { vertices[16], vertices[6], vertices[19], color, 0.0 };
	trigs[46] = { vertices[12], vertices[16], vertices[8] , color, 0.0 };
	trigs[47] = { vertices[12], vertices[7], vertices[15], color, 0.0 };
	trigs[48] = { vertices[20], vertices[12], vertices[8] , color, 0.0 };
	trigs[49] = { vertices[20], vertices[4], vertices[11], color, 0.0 };
	trigs[50] = { vertices[9], vertices[5], vertices[17], color, 0.0 };
	trigs[51] = { vertices[13], vertices[2], vertices[9] , color, 0.0 };
	trigs[52] = { vertices[17], vertices[3], vertices[13], color, 0.0 };
	trigs[53] = { vertices[2], vertices[6], vertices[10], color, 0.0 };
	trigs[54] = { vertices[3], vertices[4], vertices[14], color, 0.0 };
	trigs[55] = { vertices[5], vertices[7], vertices[18], color, 0.0 };
	trigs[56] = { vertices[6], vertices[15], vertices[5] , color, 0.0 };
	trigs[57] = { vertices[12], vertices[3], vertices[18], color, 0.0 };
	trigs[58] = { vertices[20], vertices[2], vertices[14], color, 0.0 };
	trigs[59] = { vertices[16], vertices[19], vertices[20], color, 0.0 };
	trigs[60] = { vertices[12], vertices[15], vertices[16], color, 0.0 };
	trigs[61] = { vertices[20], vertices[11], vertices[12], color, 0.0 };

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

void build_space() {
	build_floor({ -100, -100, 0 }, { -100, 100, 0 }, { 100, 100, 0 }, { 100, -100, 0 }, { 64, 64, 64, 0 }, 0.5);
	build_tetra({ 0, 0, 3 }, { 128, 0, 0, 0 }, 4);
	build_icosa({ -4, -4, 5 }, { 0, 128, 0, 0 }, 2);
	build_dodeca({ 6, -6, 3 }, { 128, 128, 0, 0 }, 2);
}

__device__ hit RayHit(vec3 pos, vec3 dir, trig* trigs)
{
	int k, k_min = -1;
	double ts_min;
	vec3 hitPos;
	vec3 normal;
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

		if (k_min == -1 || ts < ts_min) {
			k_min = k;
			ts_min = ts;
			hitPos = add({dir.x * ts, dir.y * ts, dir.z * ts}, pos);
			normal = norm(prod(e1, e2));
		}
	}

	return { hitPos, normal, k_min };
}

__device__ uchar4 ray(
	vec3 pos, vec3 dir, vec3 cameraDir,
	trig* trigs, light* lights, int lightsCount,
	int maxRecur, int curRecur = 0) {
	hit rayHit = RayHit(pos, dir, trigs);
	if (rayHit.trigId == -1)
		return { 0, 0, 0, 1 };

	vec3 rayHitPos = { rayHit.pos.x + 0.001, rayHit.pos.y + 0.001, rayHit.pos.z + 0.001 };

	double r = trigs[rayHit.trigId].reflection;
	double dotA = dot(dir, rayHit.normal);
	vec3 a = { rayHit.normal.x * 2 * dotA,
		rayHit.normal.y * 2 * dotA,
		rayHit.normal.z * 2 * dotA };
	vec3 reflDir = diff(dir, a);

	uchar4 res = {0, 0, 0, 0};

	for (int i = 0; i < lightsCount; i++)
	{
		light light = lights[i];

		vec3 lightDir = norm(diff(light.pos, rayHitPos));

		hit lightHit = RayHit(rayHitPos, lightDir, trigs);

		double4 lightColor = { 0.3, 0.3, 0.3, 0 };

		if (lightHit.trigId == -1)
		{
			double l = dot(rayHit.normal, lightDir);
			double specular = pow(dot(dir, reflDir), 32) + 1.0;

			lightColor =
			{
				Min(light.color.x * l / 255.0 * specular + lightColor.x, 1),
				Min(light.color.y * l / 255.0 * specular + lightColor.y, 1),
				Min(light.color.z * l / 255.0 * specular + lightColor.z, 1),
				light.color.w
			};
		}

		res =
		{
			Min(trigs[rayHit.trigId].color.x * lightColor.x + res.x, 255),
			Min(trigs[rayHit.trigId].color.y * lightColor.y + res.y, 255),
			Min(trigs[rayHit.trigId].color.z * lightColor.z + res.z, 255),
			res.w + 1
		};
	}

	if (r > 0.01 && curRecur < maxRecur)
	{
		uchar4 refl = ray(rayHitPos, reflDir, cameraDir, trigs, lights, lightsCount, maxRecur, curRecur + 1);

		res = { (uchar)(res.x * (1 - r) + r * refl.x),
			(uchar)(res.y * (1 - r) + r * refl.y),
			(uchar)(res.z * (1 - r) + r * refl.z),
			res.w + refl.w };
	}

	double refr_c = trigs[rayHit.trigId].refraction;

	if (refr_c < 0.99 && curRecur < maxRecur)
	{
		rayHitPos =
		{
			rayHit.pos.x + 0.01 * dir.x,
			rayHit.pos.y + 0.01 * dir.y,
			rayHit.pos.z + 0.01 * dir.z
		};
		uchar4 refr = ray(rayHitPos, dir, cameraDir, trigs, lights, lightsCount, maxRecur, curRecur + 1);

		res =
		{
			(uchar)(res.x * (1 - refr_c) + refr_c * refr.x),
			(uchar)(res.y * (1 - refr_c) + refr_c * refr.y),
			(uchar)(res.z * (1 - refr_c) + refr_c * refr.z),
			res.w + refr.w
		};
	}

	return res;
}

__global__ void render(
	vec3 pc, vec3 pv,
	int w, int h, double angle,
	uchar4* data, trig* trigs, light* lights,
	int lightsCount, int maxRecur)
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
			if (sqrtPixelRayCount > 1)
			{
				double4 res = { 0, 0, 0, 0 };
				double r = sqrtPixelRayCount;

				for (int k = 0; k < r; k++)
				{
					for (int p = 0; p < r; p++)
					{
						vec3 v = { -1.0 + dw * (i + k / r - r / 2), (-1.0 + dh * (j + p / r - r / 2)) * h / w, z };
						vec3 dir = mult(bx, by, bz, v);

						uchar4 aaRay = ray(pc, norm(dir), cameraDir, trigs, lights, lightsCount, maxRecur);
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

				data[(h - 1 - j) * w + i] = ray(pc, norm(dir), cameraDir, trigs, lights, lightsCount, maxRecur);
			}
		}
}

int main() {
	int k, w = 640, h = 480;
	char buff[256];
	uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	vec3 pc, pv;

	build_space();

	light lights[4] =
	{
		{{-6.0, -6.0, 8.0}, {0, 128, 0}},
		{{6.0, 6.0, 10.0}, {255, 0, 0}}
	};
	int lightCount = 2;

	trig* devTrigs;
	cudaMalloc(&devTrigs, sizeof(trig) * trigsCount);
	cudaMemcpy(devTrigs, trigs, sizeof(trig) * trigsCount, cudaMemcpyHostToDevice);

	light* devLights;
	cudaMalloc(&devLights, sizeof(light) * lightCount);
	cudaMemcpy(devLights, lights, sizeof(light) * lightCount, cudaMemcpyHostToDevice);

	uchar4* devData;
	cudaMalloc(&devData, sizeof(uchar4) * w * h);

	float elapsed = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	for (k = 0; k < 126; k++)
	{
		cudaEventRecord(start, 0);

		pc = vec3{ 6.0 * sin(0.05 * k), 6.0 * cos(0.05 * k), 5.0 + 2.0 * sin(0.1 * k) };
		pv = vec3{ 3.0 * sin(0.05 * k + M_PI), 3.0 * cos(0.05 * k + M_PI), 0.0 };
		render<<<dim3(32, 32), dim3(16, 16)>>>(pc, pv, w, h, 120.0, devData, devTrigs, devLights, lightCount, 3);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);

		cudaMemcpy(data, devData, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost);

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
	return 0;
}
