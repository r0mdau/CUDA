#include <stdio.h>
#include <cuda.h>
#include "dim.h"

struct cuComplexgpu {
	float r;
	float i;
	__device__ cuComplexgpu (float a, float b) : r(a), i(b){}
	__device__ float magnitude2(void) { return r * r + i * i; }
	__device__ cuComplexgpu operator*(const cuComplexgpu& a) {
		return cuComplexgpu(r * a.r - i * a.i, i * a.r + r*a.i);
	}
	__device__ cuComplexgpu operator+(const cuComplexgpu& a) {
		return cuComplexgpu(r + a.r, i + a.i);
	}
};

__device__ int juliagpu (int x, int y ) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM/2 - x)/(DIM / 2);
	float jy = scale * (float)(DIM/2 - y)/(DIM / 2);

	cuComplexgpu c(-0.8, 0.146);
	cuComplexgpu a(jx, jy);

	for (int i = 0; i < 200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;		
	}

	return 1;
}

__global__ void kernelgpu (unsigned char *ptr) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = (x + y * gridDim.x) * 4;
	
	int juliaValue = juliagpu(x, y);
	
	ptr[offset + 0] = 255 * juliaValue;
	ptr[offset + 1] = 0;
	ptr[offset + 2] = 0;
	ptr[offset + 3] = 255;
}


__global__ void kernelDamier(unsigned char *ptr)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	
	if ((x + y % 2) % 2) {
		int offset_x = (x * DIM_2 + threadIdx.x + (y * DIM_2 + threadIdx.y) * DIM) * 4;
		ptr[offset_x + 2] = 255;
	}
}


__global__ void kernelFlou(unsigned char * ptr, unsigned int * debug)
{
	__shared__ char sum[4];
	
	sum[0] = 0;
	sum[1] = 0;
	sum[2] = 0;
	sum[3] = 0;

	int x = blockIdx.x;
	int y = blockIdx.y;
	int cc = threadIdx.z;

	int index_ptr = (x * DIM_2 + threadIdx.x + (y * DIM_2 + threadIdx.y) * (gridDim.x * DIM_2)) * 4;
	int index_avg = (x + y * gridDim.x) * 4;
	
	__syncthreads();

	sum[cc] += ptr[index_ptr + cc] / (DIM_2 * DIM_2);
	
	__syncthreads();
	
	ptr[index_ptr + cc] = sum[cc];
	debug[index_avg + cc] = sum[cc];
}

__global__ void kernelFlou2(unsigned char * ptr)
{
	__shared__ int sum_r[DIM_2 * DIM_2];
	__shared__ int sum_g[DIM_2 * DIM_2];
	__shared__ int sum_b[DIM_2 * DIM_2];
	__shared__ int sum_a[DIM_2 * DIM_2];

	int x = blockIdx.x;
	int y = blockIdx.y;
	int position_thread_x = threadIdx.x;
	int position_thread_y = threadIdx.y;

	int offset_thread = position_thread_x + position_thread_y * DIM_2;
	int offset_block = x * DIM_2 + y * DIM * DIM_2;
	int offset_image = (offset_block + position_thread_x + position_thread_y * DIM) * 4;
	
	sum_r[offset_thread] = ptr[offset_image + 0];
	sum_g[offset_thread] = ptr[offset_image + 1];
	sum_b[offset_thread] = ptr[offset_image + 2];
	sum_a[offset_thread] = ptr[offset_image + 3];

	__syncthreads();

	int i = DIM_2 * DIM_2 / 2;

	while (i != 0)
	{
		if (offset_thread < i) {
			sum_r[offset_thread] = sum_r[offset_thread] + sum_r[offset_thread + i];
			sum_g[offset_thread] = sum_g[offset_thread] + sum_g[offset_thread + i];
			sum_b[offset_thread] = sum_b[offset_thread] + sum_b[offset_thread + i];
			sum_a[offset_thread] = sum_a[offset_thread] + sum_a[offset_thread + i];
		}
		__syncthreads();
		i /= 2;
	}

	ptr[offset_image + 0] = sum_r[0] / (DIM_2 * DIM_2);
	ptr[offset_image + 1] = sum_g[0] / (DIM_2 * DIM_2);
	ptr[offset_image + 2] = sum_b[0] / (DIM_2 * DIM_2);
	ptr[offset_image + 3] = sum_a[0] / (DIM_2 * DIM_2);

}

__global__ void kernelFlou3(unsigned char * ptr)
{
	__shared__ int sum[DIM_2 * DIM_2];

	int composante = threadIdx.z;
	int offset_thread = threadIdx.x + threadIdx.y * DIM_2;
	int offset_block = blockIdx.x * DIM_2 + blockIdx.y * DIM * DIM_2;
	int offset_image = (offset_block + threadIdx.x + threadIdx.y * DIM) * 4;

//	ptr[offset_image + composante] = (composante == 0 || composante == 3) ? 255 : 0;
//	return;
	sum[offset_thread] = ptr[offset_image + composante];

	__syncthreads();

	int i = DIM_2 * DIM_2 / 2;

	while (i != 0)
	{
		if (offset_thread < i)
			sum[offset_thread] += sum[offset_thread + i];
		__syncthreads();
		i /= 2;
	}

	ptr[offset_image + composante] = sum[0] / (DIM_2 * DIM_2);
}

int main()
{

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	

	CPUBitmap bitmap(DIM, DIM);
	unsigned char *dev_bitmap;
	unsigned int *dev_avg;
	unsigned int *host_avg;

	int size = (DIM * DIM) * (DIM_2 / DIM_2) * 4;
	cudaMalloc( (void**)&dev_bitmap, bitmap.image_size());
	cudaMalloc( (void**)&dev_avg, size);
	host_avg = (unsigned int*) malloc(sizeof(unsigned int) * size);

	dim3 grid(DIM, DIM);
	dim3 grid2(DIM / DIM_2, DIM / DIM_2);
	dim3 grid3(DIM_2, DIM_2, 4);
	dim3 grid4(DIM_2, DIM_2);

	// julia
	kernelgpu<<<grid, 1>>>(dev_bitmap);

	// damier
//	kernelDamier<<<grid2, grid3>>>(dev_bitmap);

	cudaEventRecord(start, 0);
	
	// flou
//	kernelFlou<<<grid2, grid3>>>(dev_bitmap, dev_avg);
	kernelFlou2<<<grid2, grid4>>>(dev_bitmap);
// 	kernelFlou3<<<grid2, grid3>>>(dev_bitmap);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);
	printf("time = %3.1f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);	
	cudaMemcpy(host_avg, dev_avg, bitmap.image_size(), cudaMemcpyDeviceToHost);	

	if (DEBUG)
	{
		int x, y, j;
		int gridx = DIM / DIM_2;
		for (int i = 0; i < gridx * gridx; i++)
		{
			y = i % gridx;
			x = (i - y) / gridx;
			printf("indice %d (%d, %d) : [", i, x, y);

			for (j = 0; j < 4; j++)
			{
				if (j > 0)
					printf(", ");
				printf("%d", host_avg[i * 4 + j]);
			}
			printf("]\n");
		}
	}

	bitmap.display_and_exit();

	cudaFree(dev_bitmap);
	cudaFree(dev_avg);
	free(host_avg);

	return 0;
}
