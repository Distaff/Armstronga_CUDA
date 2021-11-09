#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>
#include "cuda_uint128.h"	//Biblioteka autostwa Curtisa Seizerta https://github.com/curtisseizert/CUDA-uint128

#define INT128_
//#define INT64_

#ifdef INT128_
typedef uint128_t intType_t;
#endif

#ifdef INT64_
typedef uint64_t intType_t;
#endif

#ifdef INT128_
/* Wartosc bezwzgledna roznicy */
__device__ intType_t abs_dif(intType_t a, intType_t b) {
	if (a < b) return b - a;
	else return a - b;
}

/* Przeciazenie funkcji sqrt dla uint128_t. Zmodyikowany algorytm Newtona-Raphsona. Zwraca inta najbli¿szego pierwsiastkowi x */
__device__ intType_t sqrt(intType_t p) {
	intType_t x = 1;
	intType_t dif, difL, difR;
	while(1) {
		x = (x + p / x) / 2;
		dif = abs_dif(p, x*x);
		difL = abs_dif(p, (x - 1) * (x - 1));
		difR = abs_dif(p, (x + 1) * (x + 1));
		if ((dif < difL) && (dif < difR)) break;
	};

	return x;
}
#endif

/* Standardowa funkcja pow uzywa liczb zmiennoprzecinkowych i moze byc niedokladna dla duzych liczb calkowitych */
__host__ __device__ intType_t int_power(intType_t base, intType_t exponent) {
	intType_t result = 1;
	while (exponent != 0) {
		if ((exponent & 1) == 1)
			result = result * base;

		base = base * base;
		exponent >>= 1;
	}

	return result;
}

/*
Tablica zawierajaca wygenerowane uprzednio potegi cyfr - brak koniecznosci potegowania znacznie przyspiesza program
21 wierszy po 10 kolumn - zawiera od 0 do 20 potege kazdej z cyfr
Sposob ulozenia w pamieci jest istotny - obok siebie znajda sie te same potegi roznych cyfr,
co przeklada sie na wzrost wydajnosci (brak tzw. "cache misses") */
__device__ intType_t PWR_LOOKUP[21][10];


__global__ void fill_lookup() {
	for (uint_fast16_t exponent = 0; exponent < 21; exponent++)
		for (int base = 0; base < 10; base++) PWR_LOOKUP[exponent][base] = int_power(base, exponent);
}

//Uwaga - nie dziala dla 2
__device__ bool prime_check(intType_t num) {
#ifdef INT64_
	intType_t root = ceil(sqrt((double)num));	// Rzutowanie na double zmiejsza dok³adnoœæ. Przypadek szczególny przy zaokr¹gleniu w dó³? 
#endif
#ifdef INT128_
	intType_t root = sqrt(num);
#endif
	for (int i = 2; i <= root; i++) if (!(num % i)) return false;

	return true;
}

__device__ intType_t sum_p(intType_t N) {
	intType_t result = 0;
	uint_fast16_t digits[22];	//Nie uzywamy digits[0]
	uint_fast16_t digitsCtr = 0;

	while (N != 0) {
		digitsCtr++;
		digits[digitsCtr] = N % 10;
		N = N / 10;
	}

	for (uint_fast16_t i = 1; i <= digitsCtr; i++) {
		result += PWR_LOOKUP[digitsCtr][digits[i]];
	}

	return result;
}