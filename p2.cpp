#include <iostream>
#include "lapacke.h"
#include "blas.h"
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>



//HDdiff is from Stackoverflow
struct timespec HDdiff(struct timespec start, struct timespec end)
{
	struct timespec temp;
	if ((end.tv_nsec - start.tv_nsec) < 0) {
		temp.tv_sec = end.tv_sec - start.tv_sec - 1;
		temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
	}
	else {
		temp.tv_sec = end.tv_sec - start.tv_sec;
		temp.tv_nsec = end.tv_nsec - start.tv_nsec;
	}
	return temp;
}



void testLapack(double *a, double *b, int n) {
	struct timespec begin, end, diff;
	int lda = n, info = 3;
	int *ipiv = new int[n];

	clock_gettime(CLOCK_MONOTONIC, &begin);

	LAPACK_dgetrf(&n, &n, a, &lda, ipiv, &info);

	for (int i = 0; i < n; ++i) {
		double tmp = b[iptv[i] - 1];
		b[iptv[i] - 1] = b[i];
		b[i] = tmp;
	}


	char     SIDE = 'L';
	char     UPLO = 'L';
	char    TRANS = 'N';
	char     DIAG = 'U';
	double alpha = 1;

	// forward  L(Ux) = B => y = Ux
	dtrsm_(&SIDE, &UPLO, &TRANS, &DIAG, &n, &n, &alpha, A, &n, B, &n);
	UPLO = 'U';
	DIAG = 'N';
	// backward Ux = y
	dtrsm_(&SIDE, &UPLO, &TRANS, &DIAG, &n, &n, &alpha, A, &n, B, &n);

	clock_gettime(CLOCK_MONOTONIC, &end);
	diff = HDdiff(begin, end);
	printf("LAPACK, n=%d, Time:%ld seconds and %ld nanoseconds.\n", n, diff.tv_sec, diff.tv_nsec);
}





int main() {
	double *a, *b;
	int n = 10;
	srand(419);
	a = new double[n*n];
	b = new double[n];

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			a[i*n + j] = (double)((rand() << 15) | rand()) / (double)rand();
		}
		b[i] = (double)((rand() << 15) | rand()) / (double)rand();
	}

	double *ai, *bi;
	ai = new double[n*n];
	bi = new double[n];
	memcpy(ai, a, n*n * sizeof(double));
	memcpy(bi, b, n * sizeof(double));

	testLapack(ai, bi, n);


	delete[] a;
	delete[] b;
	delete[] ai;
	delete[] bi;

	return 0;
}
