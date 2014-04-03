#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 10

void addVect(void);
void firstTest(void);
void getDevicesInfos(void);

__global__ void AddInsCUDA(int *a, int *b);
__global__ void AddVectCUDA(int *a, int *b, int *c);

int main()
{
	/* First function test */
	firstTest();

	/* Select the best GPU to run the code */
	getDevicesInfos();

	/* Add vector */
	addVect();

    return 0;
}

__global__ void AddInsCUDA(int *a, int *b)
{
	//a[0] += b[0];
	*a += *b;
}

__global__ void AddVectCUDA(int *a, int *b, int *c)
{
	int tid = blockIdx.x;
	if (tid < N)
	{
		//c[tid] = a[tid] + b[tid];
		*(c + tid) = *(a + tid) + *(b + tid);
	}
}

void addVect(void)
{
	int a[N], b[N], c[N], *devA, *devB, *devC;

	cudaMalloc(&devA, N * sizeof(int));
	cudaMalloc(&devB, N * sizeof(int));
	cudaMalloc(&devC, N * sizeof(int));
	
	for (int i = 0; i < N; i++)
	{
		a[i] = -i;
		b[i] = (-i) * (-i);
	}

	cudaMemcpy(devA, &a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, &b, N * sizeof(int), cudaMemcpyHostToDevice);

	AddVectCUDA<<<N,1>>>(devA, devB, devC);

	cudaMemcpy(&c, devC, N * sizeof(int), cudaMemcpyDeviceToHost);

	printf( "\n--- Ajout de vecteurs ---\n");
	for (int i = 0; i < N; ++i)
	{
		printf("c[%d] = %d\n", i, c[i]);
	}

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);
}

void getDevicesInfos(void)
{
        int count;
 
        cudaDeviceProp prop;
        cudaGetDeviceCount( &count );
 
        for (int i = 0; i < count; ++i) {
            cudaGetDeviceProperties( &prop, i );

            printf( "\n--- Informations sur le GPU %d ---\n", i );
            printf( "Nom : %s\n", prop.name );
            printf( "Capacite de calcul : %d.%d\n", prop.major, prop.minor );
            printf( "Frequence d'horloge : %d\n", prop.clockRate );
            printf( "Total de la memoire globale : %ld\n", prop.totalGlobalMem );
 
            printf("\n--- Informations MP pour le GPU %d ---\n", i );
            printf("Nombre de multiprocesseurs : %d\n", prop.multiProcessorCount );
            printf("Memoire partagee par MP : %ld\n", prop.sharedMemPerBlock );
            printf("Registres par MP : %d\n", prop.regsPerBlock );
            printf("Threads par warp : %d\n", prop.warpSize );
            printf("Nombre maximal de threads par bloc : %d\n\n",  prop.maxThreadsPerBlock );
    }
}

void firstTest(void)
{
	int a = 5, b = 10, *da, *db;

	cudaMalloc(&da, sizeof(int));
	cudaMalloc(&db, sizeof(int));

	cudaMemcpy(da, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(db, &b, sizeof(int), cudaMemcpyHostToDevice);

	AddInsCUDA<<<1,1>>>(da, db);

	cudaMemcpy(&a, da, sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&b, db, sizeof(int), cudaMemcpyDeviceToHost);

	printf( "\n--- Premier Test ---\n");
	printf("a = %d\n", a);
	
	cudaFree(da);
	cudaFree(db);
}