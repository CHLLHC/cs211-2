#include <iostream>
#include "lapacke.h"
#include "blas.h"
#include <time.h>

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
