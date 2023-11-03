#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#define CHUNKSIZE 50

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
    unsigned long long r_sq = r*r;
    #pragma omp parallel for    \
    schedule(guided,CHUNKSIZE)      \
    reduction(+ : pixels)
        for (unsigned long long x = 0; x < r; x++) {
            unsigned long long y = ceil(sqrtl(r_sq - x*x));
            pixels += y;
        }
    pixels = pixels % k;
	printf("%llu\n", (4 * pixels) % k);
}
