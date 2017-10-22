#include <iostream>
#include "lapacke.h"
#include "blas.h"
#include <unistd.h>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

//declearation
int mydgetrf(int row, int col, double *a, int lda, int *ipiv);
int Blocked_dgetrf(int row, int col, double *a, int lda, int *ipiv, int block_size);
int mydtrsm(char trans, int n, int nrhs, double *a, int lda, int* ipiv, double *b, int ldb);

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

//Sqeuence to time the LAPACK perfermance
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

	char TRANS = 'N';
	int m = 1;
	info = LAPACKE_dgetrs(LAPACK_COL_MAJOR, TRANS, n, m, a, n, ipiv, b, n);
	if (info != 0) {
		std::cout << "LAPACKE_dgetrs FAILED" << std::endl;
	}

	clock_gettime(CLOCK_MONOTONIC, &end);
	diff = HDdiff(begin, end);
	printf("LAPACK, n=%d, Time:%ld seconds and %ld nanoseconds.\n", n, diff.tv_sec, diff.tv_nsec);
}

//Sqeuence to time the my algorithm perfermance
void testMine(double *a, double *b, int n) {
	struct timespec begin, end, diff;
	int lda = n, info = 3;
	int *ipiv = new int[n];

	clock_gettime(CLOCK_MONOTONIC, &begin);

	info = mydgetrf(n, n, a, lda, ipiv);
	if (info != 0) {
		std::cout << "mydgetrf FAILED" << std::endl;
		return;
	}

	char TRANS = 'N';
	int m = 1;
	info = mydtrsm(TRANS, n, m, a, n, ipiv, b, n);
	if (info != 0) {
		std::cout << "mydtrsm FAILED" << std::endl;
	}

	clock_gettime(CLOCK_MONOTONIC, &end);
	diff = HDdiff(begin, end);
	printf("My algorithm, n=%d, Time:%ld seconds and %ld nanoseconds.\n", n, diff.tv_sec, diff.tv_nsec);
}

//Sqeuence to time the my GEPP algorithm perfermance
void testBlcoked(double *a, double *b, int n) {
	struct timespec begin, end, diff;
	int lda = n, info = 3;
	int *ipiv = new int[n];

	clock_gettime(CLOCK_MONOTONIC, &begin);

	info = Blocked_dgetrf(n, n, a, lda, ipiv,n);
	if (info != 0) {
		std::cout << "mydgetrf FAILED" << std::endl;
		return;
	}

	char TRANS = 'N';
	int m = 1;
	//info = mydtrsm(TRANS, n, m, a, n, ipiv, b, n);
	if (info != 0) {
		std::cout << "mydtrsm FAILED" << std::endl;
	}

	clock_gettime(CLOCK_MONOTONIC, &end);
	diff = HDdiff(begin, end);
	printf("My Blocked GEPP, n=%d, Time:%ld seconds and %ld nanoseconds.\n", n, diff.tv_sec, diff.tv_nsec);
}

//Unblocked dgetrf
int mydgetrf(int row, int col, double *a, int lda, int *ipiv) {
	int n = row;
	if (n != col) {
		std::cout << "ERROR, ONLY SUPPORT REGTANGLE MATRIX" << std::endl;
		return -1;
	}
	ipiv[n - 1] = n;
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
			//save pivoting infomation in LAPACK format
			ipiv[i] = maxp + 1;
			if (maxp != i) {
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

//Blocked dgetrf
int Blocked_dgetrf(int row, int col, double *a, int lda, int *ipiv, int block_size) {
	int info = 0;
	if (row != col) {
		std::cout << "ERROR, ONLY SUPPORT REGTANGLE MATRIX" << std::endl;
		return -1;
	}
	for (int p = 0; p < row; p += block_size) {
		int pb = std::min(row - p, block_size);

		////DGETRF2
		int rowToGo = row - p;
		int colToGo = pb;
		ipiv[row - 1] = row;//p+row-1=row-1
		for (int i = p; i < row - 1; ++i) {
			int maxp = i;
			int max = abs(a[i*lda + i]);
			for (int t = i + 1; t < row; ++t) {
				if (abs(a[i*lda + t]) > max) {
					maxp = t;
					max = abs(a[i*lda + t]);
				}
			}
			if (max == 0) {
				std::cout << "LUfactoration failed: coefficient matrix is singular" << std::endl;
				return -1;
			}
			else {
				//save pivoting infomation in LAPACK format
				ipiv[i] = maxp + 1;
				if (maxp != i) {
					//swap rows
					for (int j = p; j < p + colToGo; ++j) {
						double tmp = a[j*lda + i];
						a[j*lda + i] = a[j*lda + maxp];
						a[j*lda + maxp] = tmp;
					}
				}
			}
			for (int j = i + 1; j < row; ++j) {
				a[i*lda + j] /= a[i*lda + i];
				for (int k = i + 1; k < p + colToGo; k++) {
					a[k*lda + j] -= a[i*lda + j] * a[k*lda + i];
				}
			}
		}
		////END DGETRF2








	}
	return 0;
}



//My dtrsm
//signiture based on LAPACKE_dgetrs
int mydtrsm(char trans, int n, int nrhs, double *a, int lda, int* ipiv, double *b, int ldb) {
	if (trans != 'N') {
		std::cout << "ERROR, ONLY ACCEPT N TYPE MATRIX" << std::endl;
		return -1;
	}
	if ((nrhs != 1) || (lda != ldb) || (lda != n)) {
		std::cout << "ERROR, NOT SUPPORTED." << std::endl;
		return -1;
	}

	//Forward Substitution
	//Preprocess
	for (int i = 0; i < n; ++i) {
		double temp = b[ipiv[i] - 1];
		b[ipiv[i] - 1] = b[i];
		b[i] = temp;
	}
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < i; ++j) {
			b[i] -= b[j] * a[j*n + i];
		}
	}

	//Backward Substitution
	for (int i = n - 1; i >= 0; --i) {
		for (int j = i + 1; j < n; j++) {
			b[i] -= b[j] * a[j*n + i];
		}
		b[i] /= a[i*n + i];
	}
	return 0;
}


int main(int argc, char *argv[]) {
	double *a, *b;
	int n = 0, opt;

	while ((opt = getopt(argc, argv, "n:")) != EOF) {
		switch (opt) {
		case 'n':
			n = atoi(optarg);
			break;
		case '?':
		default:
			std::cerr << "Usage run -n <size>" << std::endl;
			return -1;
		}
	}
	if (n == 0) {
		std::cerr << "Usage run -n <size>" << std::endl;
		return -1;
	}

	srand(419);
	a = new double[n*n];
	b = new double[n];

	n = 9;
	a[0] = 1;
	a[1] = 2;
	a[2] = 1;
	a[3] = 2;
	a[4] = 1;
	a[5] = 1;
	a[6] = 3;
	a[7] = 1; 
	a[8] = 1;
	b[0] = 1;
	b[1] = 1;
	b[2] = 1;

	/*
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			a[i*n + j] = (double)((rand() << 15) | rand()) / (double)rand();
		}
		b[i] = (double)((rand() << 15) | rand()) / (double)rand();
	}*/

	double *al, *bl, *ag, *bg;
	al = new double[n*n];
	bl = new double[n];
	ag = new double[n*n];
	bg = new double[n];
	memcpy(al, a, n*n * sizeof(double));
	memcpy(bl, b, n * sizeof(double));
	memcpy(ag, a, n*n * sizeof(double));
	memcpy(bg, b, n * sizeof(double));

	testLapack(al, bl, n);
	testMine(a, b, n);
	testBlcoked(ag, bg, n);


	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			std::cout << a[j*n + i] << " ";
		}
		std::cout << std::endl;
	}

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			std::cout << ag[j*n + i] << " ";
		}
		std::cout << std::endl;
	}

	double sumOfSquare = 0;
	for (int i = 0; i < n; ++i) {
		sumOfSquare += (b[i] - bl[i])*(b[i] - bl[i]);
	}
	double norm = sqrt(sumOfSquare);
	std::cout << "The norm of difference between LAPACK and My unoptimized algorithm is " << std::scientific << norm << std::endl;

	sumOfSquare = 0;
	for (int i = 0; i < n; ++i) {
		sumOfSquare += (bg[i] - bl[i])*(bg[i] - bl[i]);
	}
	norm = sqrt(sumOfSquare);
	std::cout << "The norm of difference between LAPACK and My OPTIMIZED algorithm is " << std::scientific << norm << std::endl;

	delete[] a;
	delete[] b;
	delete[] al;
	delete[] bl;
	delete[] ag;
	delete[] bg;

	return 0;
}
