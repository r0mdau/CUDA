#include "water.h"
#define DIM 512

// Libère la mémoire allouée sur le GPU  
void cleanup( DataBlock *d ) {  
	cudaFree( d->dev_bitmap );  
}

void cleanupcpu( DataBlock *d ) {  
	free(d->dev_bitmap); 
}

 __global__ void kernel( unsigned char *ptr, int ticks ) {
	// Fait correspondre threadIdx/BlockIdx à des positions de pixels
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// Calcule la valeur à cette position
	float fx = x - DIM/2;
	float fy = y - DIM/2;
	float d = sqrtf( fx * fx + fy * fy );

	unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d/10.0f - ticks/7.0f) / (d/10.0f + 1.0f));

	ptr[offset*4 + 0] = grey;
	ptr[offset*4 + 1] = grey;
	ptr[offset*4 + 2] = grey;
	ptr[offset*4 + 3] = 255;
}

void generate_frame( DataBlock *d, int ticks ) {  
	long int before = GetTickCount();
	dim3 blocks(DIM/16,DIM/16);  
	dim3 threads(16,16);  

	kernel<<<blocks,threads>>>( d->dev_bitmap, ticks );  

	cudaMemcpy( d->bitmap->get_ptr(),  
		d->dev_bitmap,
		d->bitmap->image_size(),
		cudaMemcpyDeviceToHost 
	);  

	long int after = GetTickCount();
	printf("ticks %d time %ld\n", ticks, after - before);
}

void generate_framecpu( DataBlock *d, int ticks ) {  
	long int before = GetTickCount();
	unsigned char *ptr = d->bitmap->get_ptr();
	
	for(int y = 0; y < DIM ; y++) 
		for(int x = 0; x < DIM ; x++) 
		{
			int offset = x + y * DIM;			

			// Calcule la valeur à cette position
			float fx = x - DIM/2;
			float fy = y - DIM/2;
			float d = sqrtf( fx * fx + fy * fy );

			unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d/10.0f - ticks/7.0f) / (d/10.0f + 1.0f));
			
			ptr[offset*4 + 0] = grey;
			ptr[offset*4 + 1] = grey;
			ptr[offset*4 + 2] = grey;
			ptr[offset*4 + 3] = 255;
		}

	long int after = GetTickCount();
	printf("ticks %d time %ld\n", ticks, after - before);
}

int mainwatergpu( void ) {  
	DataBlock data;  
	CPUAnimBitmap bitmap( DIM, DIM, &data );  
	data.bitmap = &bitmap;  
	
	cudaMalloc( (void**)&data.dev_bitmap,  bitmap.image_size() ) ;  
	bitmap.anim_and_exit( (void (*)(void*,int))generate_frame, (void (*)(void*))cleanup );  
	
	return 0;
}

int mainwatercpu( void ) {  
	DataBlock data;  
	CPUAnimBitmap bitmap( DIM, DIM, &data );  
	data.bitmap = &bitmap;  

	data.dev_bitmap = (unsigned char *) malloc(bitmap.image_size() );
	bitmap.anim_and_exit( (void (*)(void*,int))generate_framecpu, (void (*)(void*))cleanupcpu );  

	return 0;
}