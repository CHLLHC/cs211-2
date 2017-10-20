#include <iostream>
#include "lapacke.h"
#include "blas.h"
#include <ctime>
#include <cstdio>



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



void testLapack() {
	struct timespec begin, end, diff;

	clock_gettime(CLOCK_MONOTONIC, &begin);


	clock_gettime(CLOCK_MONOTONIC, &end);
	diff = HDdiff(begin, end);
	printf("Blocked cache and register ijk, n=%d, B=%d, Time:%ld seconds and %ld nanoseconds.\n", n, B, diff.tv_sec, diff.tv_nsec);
}





int main() {

	return 0;
}
