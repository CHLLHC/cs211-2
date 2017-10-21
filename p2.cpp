#include <iostream>
#include "lapacke.h"
#include "blas.h"
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

int mydgetrf(int row, int col, double *a, int lda, int *ipiv);


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

	info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a, lda, ipiv);
	if (info != 0) {
		std::cout << "LAPACKE_dgetrf FAILED" << std::endl;
		return;
	}
	for (int i = 0; i < n; ++i) {
		std::cout << ipiv[i] << " " << std::endl;
	}
	char TRANS = 'N';
	int m = 1;
	//info = LAPACKE_dgetrs(LAPACK_COL_MAJOR, TRANS, n, m, a, n, ipiv, b, n);
	if (info != 0) {
		std::cout << "LAPACKE_dgetrs FAILED" << std::endl;
	}

	clock_gettime(CLOCK_MONOTONIC, &end);
	diff = HDdiff(begin, end);
	printf("LAPACK, n=%d, Time:%ld seconds and %ld nanoseconds.\n", n, diff.tv_sec, diff.tv_nsec);
}

void testMine(double *a, double *b, int n) {
	struct timespec begin, end, diff;
	int lda = n, info = 3;
	int *ipiv = new int[n];

	clock_gettime(CLOCK_MONOTONIC, &begin);

	info = mydgetrf(n, n, a, lda, ipiv);
	if (info != 0) {
		std::cout << "LAPACKE_dgetrf FAILED" << std::endl;
		return;
	}
	for (int i = 0; i < n; ++i) {
		std::cout << ipiv[i] << " " << std::endl;
	}
	char TRANS = 'N';
	int m = 1;
	//info = LAPACKE_dgetrs(LAPACK_COL_MAJOR, TRANS, n, m, a, n, ipiv, b, n);
	if (info != 0) {
		std::cout << "LAPACKE_dgetrs FAILED" << std::endl;
	}

	clock_gettime(CLOCK_MONOTONIC, &end);
	diff = HDdiff(begin, end);
	printf("MyFunction, n=%d, Time:%ld seconds and %ld nanoseconds.\n", n, diff.tv_sec, diff.tv_nsec);
}



int mydgetrf(int row, int col, double *a, int lda, int *ipiv) {

	int n = row;
	if (n != col) {
		std::cout << "ERROR, ONLY SUPPORT REGTANGLE MATRIX" << std::endl;
		return -1;
	}
	
	for (int i = 0; i < n - 1; ++i) {
		int maxp = i;
		int max = abs(a[i*n + i]);
		for (int t = i + 1; t < n; ++t) {
			if (abs(a[i*n + t]) > max) {
				maxp = t;
				max = abs(a[i*n + t]);
			}
		}
		if (max == 0) {
			std::cout << "LUfactoration failed: coefficient matrix is singular" << std::endl;
			return -1;
		}
		else {
			ipiv[i] = maxp;
			if (maxp != i) {
				//save pivoting infomation
				/*
				int temp = ipiv[i];
				ipiv[i] = ipiv[maxp];
				ipiv[maxp] = temp;
				*/
				//swap rows
				for (int j = 0; j < n; ++j) {
					double tmp = a[j*n + i];
					a[j*n + i] = a[j*n + maxp];
					a[j*n + maxp] = tmp;
				}
			}
		}
		for (int j = i + 1; j < n; ++j) {
			a[i*n + j] /= a[i*n + i];
			for (int k = i + 1; k < n; k++) {
				a[k*n + j] -= a[i*n + j] * a[k*n + i];
			}
		}
	}
	return 0;
}


int main() {
	//double *a, *b;
	int n = 3;
	srand(419);
	//a = new double[n*n];
	//b = new double[n];

	double  a[9] =
	{
		1, 2, 1,
		2, 1, 1,
		3, 1, 1
	};

	double b[3] =
	{
		1,
		1,
		1
	};



	/*for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			a[i*n + j] = (double)((rand() << 15) | rand()) / (double)rand();
		}
		b[i] = (double)((rand() << 15) | rand()) / (double)rand();
	}*/

	double *al, *bl;
	al = new double[n*n];
	bl = new double[n];
	memcpy(al, a, n*n * sizeof(double));
	memcpy(bl, b, n * sizeof(double));

	testLapack(al, bl, n);
	/*for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			std::cout << al[i*n + j] << " ";
		}
		std::cout << std::endl;
	}*/

	testMine(a, b, n);


	//delete[] a;
	//delete[] b;
	delete[] al;
	delete[] bl;

	return 0;
}
