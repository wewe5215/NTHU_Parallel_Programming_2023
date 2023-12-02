#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <algorithm>
#include <pthread.h>
#include <iostream>
cpu_set_t cpu_set;
int ncpus;
int* d;
int** d_tmp;
int V, E;
int V_sq;
pthread_barrier_t barrier;
// If there is no valid path between i->j, dist(i, j) = 2 ^ 30 − 1 = 1073741823.
const int MAXIMUM = ((1 << 30) - 1);
void handle_input(char* input_file){
    FILE* file = fopen(input_file, "rb");

    // Read the number of vertices and edges
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);
    V_sq = V * V;
    d = (int*)malloc((V_sq + 5) * sizeof(int));
    d_tmp = (int**)malloc(V * sizeof(int*));
    int i;
    for(i = 0; i < V; i ++){
        d_tmp[i] = d + i * V;
        for(int j = 0; j < V; j ++){
            if(i == j)d_tmp[i][j] = 0;
            else d_tmp[i][j] = MAXIMUM;
        }
    }

    for(i = 0; i < E; i ++){
        int src, dst, dist;
        fread(&src, sizeof(int), 1, file);
        fread(&dst, sizeof(int), 1, file);
        fread(&dist, sizeof(int), 1, file);
        d_tmp[src][dst] = dist;
    }

    fclose(file);
    return;
}

void* Floyd_Warshall(void* threadid){
    int tid = *(int*) threadid;
    int i, j, k;
    for(k = 0; k < V; k ++){
        for(i = tid; i < V; i += ncpus){
            for(j = 0; j < V; j ++){
                d_tmp[i][j] = std::min(d_tmp[i][j], d_tmp[i][k] + d_tmp[k][j]);
            }
        }
        pthread_barrier_wait(&barrier);
    }
    return NULL;
}

void handle_output(char* output_file){
    FILE *file = fopen(output_file, "w");
    fwrite(d, sizeof(int), V_sq, file);
    fclose(file);
    return;
}
int main(int argc, char** argv) {

    /* detect how many CPUs are available */
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    ncpus = CPU_COUNT(&cpu_set);
    pthread_t threads[ncpus];
    int thread_ids[ncpus];
    pthread_barrier_init(&barrier, NULL, ncpus);
    handle_input(argv[1]);
    // Floyd_Warshall();
    int t, rc;
    for (t = 0; t < ncpus; t++) {
        thread_ids[t] = t;
        rc = pthread_create(&threads[t], NULL, Floyd_Warshall, (void*)&thread_ids[t]);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }
	for(t = 0; t < ncpus; t++)
		pthread_join(threads[t], NULL);
    pthread_barrier_destroy(&barrier);
    handle_output(argv[2]);
    free(d);
}