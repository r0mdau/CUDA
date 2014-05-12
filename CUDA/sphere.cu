#include <iostream>
#include <stdlib.h>
#include <time.h>
#include "common\cpu_bitmap.h"
#include "common\book.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sphere.h"

#define DIM 512
#define NUM_SPHERES 30

struct cuComplex {
	float r;
	float i;
	__device__ cuComplex (float a, float b) : r(a), i(b){}
	__device__ float magnitude2(void) { return r * r + i * i; }
	__device__ cuComplex operator*(const cuComplex& a) {
		return cuComplex(r * a.r - i * a.i, i * a.r + r*a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}
};

__device__ int julia (int x, int y ) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM/2 - x)/(DIM / 2);
	float jy = scale * (float)(DIM/2 - y)/(DIM / 2);

	cuComplex c(-0.9, 0.156);
	cuComplex a(jx, jy);

	for (int i = 0; i < 200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;		
	}

	return 1;
}

struct Sphere{
	int r, g ,b;
	float rayon;
	float x, y, z;
	void init(int or, int og, int ob, float ox, float oy, float oz, float orayon){
		r = or; g = og; b = ob;
		x = ox; y = oy; z = oz;
		rayon = orayon;
	}
	__device__ float touche(float ox, float oy){
		float distance = sqrt(pow((ox - x), 2) + pow((oy - y), 2));
		if(distance < rayon){
			return distance;
		}else return 0;
	}
};

__global__ void kernel (Sphere * tab, unsigned char *ptr) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float ox = x - DIM / 2;
	float oy = y - DIM / 2;

	ptr[offset * 4 + 0] = 0;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;

	for(int i = 0; i < NUM_SPHERES; i++){
		float distance = tab[i].touche(ox, oy);
		if(distance){
			float attenuation = 1 - (distance / tab[i].rayon);
			ptr[offset * 4 + 0] = tab[i].r * attenuation + attenuation * 10;
			ptr[offset * 4 + 1] = tab[i].g * attenuation + attenuation * 10;
			ptr[offset * 4 + 2] = tab[i].b * attenuation + attenuation * 10;
			ptr[offset * 4 + 3] = 255;
		}
	}
}

void mainSphere (void) {
	srand(time(NULL));
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *dev_bitmap;
	Sphere *scene;

	struct Sphere sphere1, sphere2, sphere3;
	Sphere tab[NUM_SPHERES];
	for(int i = 0; i < NUM_SPHERES; i++){
		tab[i].init(rand() % 256, rand() % 256, rand() % 256, rand() % DIM, rand() % DIM, rand() % DIM, 50);
	}

	cudaMalloc( (void**)&scene, sizeof(Sphere) * NUM_SPHERES);
	cudaMemcpy(scene, tab, sizeof(Sphere) * NUM_SPHERES, cudaMemcpyHostToDevice);

	cudaMalloc( (void**)&dev_bitmap, bitmap.image_size());

	dim3 grid(DIM/16, DIM/16);
	dim3 threads(16, 16);
	kernel<<<grid, threads>>>(scene, dev_bitmap);

	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);	

	bitmap.display_and_exit();

	cudaFree(dev_bitmap);
}