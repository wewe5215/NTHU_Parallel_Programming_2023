#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <algorithm>
#include <pthread.h>
#include <iostream>
#include <zlib.h>
#include <cstdlib>
#include <cassert>
#include <time.h>
using namespace std;

#define blocksize 64
// phase2 need 3 arr --> 49152 / 4(int) / 3 = 4096
// arr is square --> 4096 = 64 * 64 --> blocksize = 64
// (64 * 64) / 1024(number of threads per block) = 4 per thread
// each thread is responsible for 4 data
int* d;
int V, E;
int V_sq;
int original_V;
// If there is no valid path between i->j, dist(i, j) = 2 ^ 30 − 1 = 1073741823.
const int MAXIMUM = ((1 << 30) - 1);


void handle_input(char* input_file){
    FILE* file = fopen(input_file, "rb");

    // Read the number of vertices and edges
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);
    // for blocksize = 64
    original_V = V;
    if(V % blocksize != 0){
        V += (blocksize - V % blocksize);
    }
    V_sq = V * V;
    d = (int*)malloc(V_sq * sizeof(int));
    int i;
    for(i = 0; i < V; i ++){
        for(int j = 0; j < V; j ++){
            int idx = i * V + j;
            if(i == j)d[idx] = 0;
            else d[idx] = MAXIMUM;
        }
    }

    for(i = 0; i < E; i ++){
        int src, dst, dist;
        fread(&src, sizeof(int), 1, file);
        fread(&dst, sizeof(int), 1, file);
        fread(&dist, sizeof(int), 1, file);
        d[src * V + dst] = dist;
    }

    fclose(file);
    return;
}

__global__ void Phase1(int* d, int round, int V){
    int i = threadIdx.x;
    int j = threadIdx.y;
    // real index in d
    int idx_x = i + round * blocksize;
    int idx_y = j + round * blocksize;
    int idx_d = idx_y * V + idx_x;
    // calculation
    for(int k = 0; k < blocksize; k ++){
        int idx1 = idx_y * V + 32 * V + (k + round * blocksize);//[j + 32][k]
        int idx2 = idx_y * V + (k + round * blocksize);//[j][k]
        int idx3 = (k + round * blocksize) * V + idx_x;//[k][i]
        int idx4 = (k + round * blocksize) * V + idx_x + 32;//[k][i + 32]
        d[idx_d] = (d[idx_d] > d[idx2] + d[idx3]) ? d[idx2] + d[idx3] : d[idx_d];
        d[idx_d + V * 32] = (d[idx_d + V * 32] > d[idx1] + d[idx3]) ? d[idx1] + d[idx3] : d[idx_d + V * 32];
        d[idx_d + 32] = (d[idx_d + 32] > d[idx2] + d[idx4]) ? d[idx2] + d[idx4] : d[idx_d + 32];
        d[idx_d + V * 32 + 32] = (d[idx_d + V * 32 + 32] > d[idx1] + d[idx4]) ? d[idx1] + d[idx4] : d[idx_d + V * 32 + 32];
        __syncthreads();
    }
}

__global__ void Phase2(int* d, int round, int V){
    if(round == blockIdx.y)return;
    int i = threadIdx.x;
    int j = threadIdx.y;
    // real index in d
    int idx_x = i + round * blocksize;
    int idx_y = j + round * blocksize;
    int idx_x_row = i + blockIdx.y * blocksize; //y is fixed, only x change
    int idx_y_col = j + blockIdx.y * blocksize; //x is fixed, only y change
    // pivot
    int idx_d = idx_y * V + idx_x;
    int idx_row_d = idx_y * V + idx_x_row;
    int idx_col_d = idx_y_col * V + idx_x;
    for(int k = 0; k < blocksize; k ++){
        int idx1 = idx_y * V + 32 * V + (k + round * blocksize);//[j + 32][k]
        int idx2 = idx_y * V + (k + round * blocksize);//[j][k]
        int idx3 = (k + round * blocksize) * V + idx_x_row;//[k][i]
        int idx4 = (k + round * blocksize) * V + idx_x_row + 32;//[k][i + 32]
        d[idx_row_d] = (d[idx_row_d] > d[idx2] + d[idx3]) ? d[idx2] + d[idx3] : d[idx_row_d];
        d[idx_row_d + V * 32] = (d[idx_row_d + V * 32] > d[idx1] + d[idx3]) ? d[idx1] + d[idx3] : d[idx_row_d + V * 32];
        d[idx_row_d + 32] = (d[idx_row_d + 32] > d[idx2] + d[idx4]) ? d[idx2] + d[idx4] : d[idx_row_d + 32];
        d[idx_row_d + V * 32 + 32] = (d[idx_row_d + V * 32 + 32] > d[idx1] + d[idx4]) ? d[idx1] + d[idx4] : d[idx_row_d + V * 32 + 32];

        idx1 = idx_y_col * V + 32 * V + (k + round * blocksize);//[j + 32][k]
        idx2 = idx_y_col * V + (k + round * blocksize);//[j][k]
        idx3 = (k + round * blocksize) * V + idx_x;//[k][i]
        idx4 = (k + round * blocksize) * V + idx_x + 32;//[k][i + 32]
        d[idx_col_d] = (d[idx_col_d] > d[idx2] + d[idx3]) ? d[idx2] + d[idx3] : d[idx_col_d];
        d[idx_col_d + V * 32] = (d[idx_col_d + V * 32] > d[idx1] + d[idx3]) ? d[idx1] + d[idx3] : d[idx_col_d + V * 32];
        d[idx_col_d + 32] = (d[idx_col_d + 32] > d[idx2] + d[idx4]) ? d[idx2] + d[idx4] : d[idx_col_d + 32];
        d[idx_col_d + V * 32 + 32] = (d[idx_col_d + V * 32 + 32] > d[idx1] + d[idx4]) ? d[idx1] + d[idx4] : d[idx_col_d + V * 32 + 32];
        __syncthreads();
    }
}

__global__ void Phase3(int* d, int round, int V){
    // put data to shared memory
    if(round == blockIdx.x || round == blockIdx.y)return;
    __shared__ int pivot[blocksize][blocksize];
    __shared__ int row[blocksize][blocksize];
    __shared__ int col[blocksize][blocksize];
    int i = threadIdx.x;
    int j = threadIdx.y;
    // real index in d
    int idx_x = i + round * blocksize;
    int idx_y = j + round * blocksize;
    int idx_x_row = i + blockIdx.x * blocksize; //y is fixed, only x change
    int idx_y_col = j + blockIdx.y * blocksize; //x is fixed, only y change
    // pivot
    int idx_d = idx_y_col * V + idx_x_row;
    int idx_row_d = idx_y_col * V + idx_x;
    int idx_col_d = idx_y * V + idx_x_row;
    for(int k = 0; k < blocksize; k ++){
        int idx1 = idx_y_col * V + 32 * V + (k + round * blocksize);//[j + 32][k]
        int idx2 = idx_y_col * V + (k + round * blocksize);//[j][k]
        int idx3 = (k + round * blocksize) * V + idx_x_row;//[k][i]
        int idx4 = (k + round * blocksize) * V + idx_x_row + 32;//[k][i + 32]
        d[idx_d] = (d[idx_d] > d[idx2] + d[idx3]) ? d[idx2] + d[idx3] : d[idx_d];
        d[idx_d + V * 32] = (d[idx_d + V * 32] > d[idx1] + d[idx3]) ? d[idx1] + d[idx3] : d[idx_d + V * 32];
        d[idx_d + 32] = (d[idx_d + 32] > d[idx2] + d[idx4]) ? d[idx2] + d[idx4] : d[idx_d + 32];
        d[idx_d + V * 32 + 32] = (d[idx_d + V * 32 + 32] > d[idx1] + d[idx4]) ? d[idx1] + d[idx4] : d[idx_d + V * 32 + 32];
        __syncthreads();
    }

}



void handle_output(char* output_file){
    FILE *file = fopen(output_file, "w");
    for(int i = 0; i < original_V; i ++){
        fwrite(d + i * V, sizeof(int), original_V, file);
    }
    fclose(file);
    return;
}
int main(int argc, char** argv) {

    handle_input(argv[1]);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    // for GPU code
    size_t d_size = V_sq * sizeof(int);// use size_t instead of int !!!
    cudaHostRegister(d, d_size, cudaHostRegisterDefault);
    int* device_d;
    cudaMalloc(&device_d, d_size);
    cudaMemcpy(device_d, d, d_size, cudaMemcpyHostToDevice);
    dim3 blk(32, 32);
    int total_rnd = V / blocksize;
    dim3 grid_ph2(1, total_rnd);
    dim3 grid_ph3(total_rnd, total_rnd);
    for(int k = 0; k < total_rnd; k ++){
        //dot
        Phase1 << <1, blk>>> (device_d, k, V);
        //line
        Phase2 << <grid_ph2, blk>>>(device_d, k, V);
        //plane
        Phase3 << <grid_ph3, blk>>>(device_d, k, V);
    }

    cudaMemcpy(d, device_d, d_size, cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Elapsed time: %.9f seconds\n", elapsed);
    handle_output(argv[2]);
    return 0;
}
