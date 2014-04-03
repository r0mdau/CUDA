#include <stdio.h>
#include "julia.h"

#define DIM 800

struct cuComplex
{
	float r, i;

	cuComplex(float a, float b) : r(a), i(b) {}

	cuComplex operator*(const cuComplex &a) {
		return cuComplex(r * a.r, i * a.i);
	}

	cuComplex operator+(const cuComplex &a) {
		return cuComplex(r + a.r, i + a.i);
	}
};