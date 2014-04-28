#ifndef _DIM_H_

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <device_functions.h>

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common\cpu_bitmap.h"
#include "common\book.h"

#define DIM 400
#define DIM_2 4	
#define DEBUG false

#endif
