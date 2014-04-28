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


__global__ void kernelFlou(unsigned char * ptr, unsigned int * avg)
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
	// avg[index_avg] = 0;
	
	__syncthreads();

	sum[cc] += ptr[index_ptr + cc] / (DIM_2 * DIM_2);
	
	__syncthreads();
	
	avg[index_avg + cc] = sum[cc];
	ptr[index_ptr + cc] = sum[cc];
}

int main()
{

	CPUBitmap bitmap(DIM, DIM);
	unsigned char *dev_bitmap;
	unsigned int *dev_avg;
	unsigned int *host_avg;

	int size = (DIM * DIM) * (DIM_2 / DIM_2) * 4;
	cudaMalloc( (void**)&dev_bitmap, bitmap.image_size());
	cudaMalloc( (void**)&dev_avg, size);
	host_avg = (unsigned int*) malloc(sizeof(unsigned int) * size);

	dim3 grid(DIM, DIM);
	// julia
	kernelgpu<<<grid, 1>>>(dev_bitmap);

	dim3 grid2(DIM / DIM_2, DIM / DIM_2);
	dim3 grid3(DIM_2, DIM_2, 4);

	// damier
	kernelDamier<<<grid2, grid3>>>(dev_bitmap);

	// flou
	kernelFlou<<<grid2, grid3>>>(dev_bitmap, dev_avg);

	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);	
	cudaMemcpy(host_avg, dev_avg, bitmap.image_size(), cudaMemcpyDeviceToHost);	

	if (DEBUG)
	{
		// i = (x + y * DIM / TOTO) * 4
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