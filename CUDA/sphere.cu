#include <iostream>
#include <stdlib.h>
#include <time.h>
#include "common\cpu_bitmap.h"
#include "common\book.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sphere.h"

#define DIM 512
#define NUM_SPHERES 300

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

	float ox = x;
	float oy = y;

	ptr[offset * 4 + 0] = 255;
	ptr[offset * 4 + 1] = 255;
	ptr[offset * 4 + 2] = 255;
	ptr[offset * 4 + 3] = 255;

	for(int i = 0; i < NUM_SPHERES; i++){
		float distance = tab[i].touche(ox, oy);
		if(distance){
			float attenuation = 1 - (distance / tab[i].rayon / 1.2);
			ptr[offset * 4 + 0] = tab[i].r * attenuation;
			ptr[offset * 4 + 1] = tab[i].g * attenuation;
			ptr[offset * 4 + 2] = tab[i].b * attenuation;
			ptr[offset * 4 + 3] = 255;
		}
	}
}

void mainSphere (void) {
	srand(time(NULL));
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *dev_bitmap;
	Sphere *scene;

	Sphere tab[NUM_SPHERES];
	for(int i = 0; i < NUM_SPHERES; i++){
		tab[i].init(rand() % 256, rand() % 256, rand() % 256, rand() % DIM, rand() % DIM, rand() % DIM, rand () % 50);
	}

	cudaMalloc( (void**)&scene, sizeof(Sphere) * NUM_SPHERES);
	cudaMemcpy(scene, tab, sizeof(Sphere) * NUM_SPHERES, cudaMemcpyHostToDevice);

	cudaMalloc( (void**)&dev_bitmap, bitmap.image_size());

	dim3 grid(DIM/16, DIM/16);
	dim3 threads(16, 16);
	
	// DEBUT calcul du temps d'exécution du floutage
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	kernel<<<grid, threads>>>(scene, dev_bitmap);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf(" Temps de generation %3.1f ms\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(start);
	// FIN calcul du temps d'exécution du floutage

	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);	

	bitmap.display_and_exit();

	cudaFree(dev_bitmap);
	cudaFree(scene);
}