#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
unsigned long long r, k;
unsigned long long pixels = 0;
unsigned long long ncpus = 0;
unsigned long long r_sq = 0;
void* computing(void* threadid){
	unsigned long long local_pixels = 0;
	unsigned long long tid = *(int*)threadid;
	unsigned long long elem_num = r / ncpus;
	unsigned long long upper_bound = (tid == ncpus - 1) ? r : (tid + 1) * elem_num;
	for(unsigned long long x = tid * elem_num; x < upper_bound; x ++){
		unsigned long long y = ceil(sqrtl(r_sq - x*x));
		local_pixels += y;
		local_pixels %= k;
	}
	pixels += local_pixels;
	pthread_exit(NULL);
}
int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	r = atoll(argv[1]);
	k = atoll(argv[2]);
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ncpus = CPU_COUNT(&cpuset);
    pthread_t threads[ncpus];
    int rc;
    int ID[ncpus];
    int t;
	r_sq = r * r;
	for (t = 0; t < ncpus; t++) {
        ID[t] = t;
        rc = pthread_create(&threads[t], NULL, computing, (void*)&ID[t]);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }
	for(t = 0; t < ncpus; t++)
		pthread_join(threads[t], NULL);
	printf("%llu\n", (4 * pixels) % k);
}