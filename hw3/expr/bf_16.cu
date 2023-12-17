#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <algorithm>
#include <pthread.h>
#include <iostream>
#include <zlib.h>
#include <cstdlib>
#include <cassert>
using namespace std;

#define blocksize 16
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
    // put data to shared memory
    __shared__ int shared[blocksize][blocksize];
    int i = threadIdx.x;
    int j = threadIdx.y;
    // real index in d
    int idx_x = i + round * blocksize;
    int idx_y = j + round * blocksize;
    int idx_d = idx_y * V + idx_x;
    shared[j][i] = d[idx_d];
    shared[j + 8][i] = d[idx_d + V * 8];
    shared[j][i + 8] = d[idx_d + 8];
    shared[j + 8][i + 8] = d[idx_d + V * 8 + 8];
    __syncthreads();
    // calculation
    #pragma unroll
    for(int k = 0; k < blocksize; k ++){
        shared[j][i] = min(shared[j][i], shared[j][k] + shared[k][i]);
        shared[j + 8][i] = min(shared[j + 8][i], shared[j + 8][k] + shared[k][i]);
        shared[j][i + 8] = min(shared[j][i + 8], shared[j][k] + shared[k][i + 8]);
        shared[j + 8][i + 8] = min(shared[j + 8][i + 8], shared[j + 8][k] + shared[k][i + 8]);
        __syncthreads();
    }

    //write back
    d[idx_d] = shared[j][i];
    d[idx_d + V * 8] = shared[j + 8][i];
    d[idx_d + 8] = shared[j][i + 8];
    d[idx_d + V * 8 + 8] = shared[j + 8][i + 8];
}

__global__ void Phase2(int* d, int round, int V){
    if(round == blockIdx.y)return;
    // put data to shared memory
    __shared__ int pivot[blocksize][blocksize];
    __shared__ int row[blocksize][blocksize];
    __shared__ int col[blocksize][blocksize];
    int i = threadIdx.x;
    int j = threadIdx.y;
    // real index in d
    int idx_x = i + round * blocksize;
    int idx_y = j + round * blocksize;
    int idx_x_row = i + blockIdx.y * blocksize; //y is fixed, only x change
    int idx_y_col = j + blockIdx.y * blocksize; //x is fixed, only y change
    // pivot
    int idx_d = idx_y * V + idx_x;
    pivot[j][i] = d[idx_d];
    pivot[j + 8][i] = d[idx_d + V * 8];
    pivot[j][i + 8] = d[idx_d + 8];
    pivot[j + 8][i + 8] = d[idx_d + V * 8 + 8];
    //row --> y fixed
    int idx_row_d = idx_y * V + idx_x_row;
    row[j][i] = d[idx_row_d];
    row[j + 8][i] = d[idx_row_d + V * 8];
    row[j][i + 8] = d[idx_row_d + 8];
    row[j + 8][i + 8] = d[idx_row_d + V * 8 + 8];
    //col --> x fixed
    int idx_col_d = idx_y_col * V + idx_x;
    col[j][i] = d[idx_col_d];
    col[j + 8][i] = d[idx_col_d + V * 8];
    col[j][i + 8] = d[idx_col_d + 8];
    col[j + 8][i + 8] = d[idx_col_d + V * 8 + 8];
    __syncthreads();
    #pragma unroll
    for(int k = 0; k < blocksize; k ++){
        row[j][i] = min(row[j][i], pivot[j][k] + row[k][i]);
        row[j + 8][i] = min(row[j + 8][i], pivot[j + 8][k] + row[k][i]);
        row[j][i + 8] = min(row[j][i + 8], pivot[j][k] + row[k][i + 8]);
        row[j + 8][i + 8] = min(row[j + 8][i + 8], pivot[j + 8][k] + row[k][i + 8]);

        col[j][i] = min(col[j][i], col[j][k] + pivot[k][i]);
        col[j + 8][i] = min(col[j + 8][i], col[j + 8][k] + pivot[k][i]);
        col[j][i + 8] = min(col[j][i + 8], col[j][k] + pivot[k][i + 8]);
        col[j + 8][i + 8] = min(col[j + 8][i + 8], col[j + 8][k] + pivot[k][i + 8]);
        __syncthreads();
    }

    //row --> y fixed
    d[idx_row_d] = row[j][i];
    d[idx_row_d + V * 8] = row[j + 8][i];
    d[idx_row_d + 8] = row[j][i + 8];
    d[idx_row_d + V * 8 + 8] = row[j + 8][i + 8];
    //col --> x fixed
    d[idx_col_d] = col[j][i];
    d[idx_col_d + V * 8] = col[j + 8][i];
    d[idx_col_d + 8] = col[j][i + 8];
    d[idx_col_d + V * 8 + 8] = col[j + 8][i + 8];
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
    pivot[j][i] = d[idx_d];
    pivot[j + 8][i] = d[idx_d + V * 8];
    pivot[j][i + 8] = d[idx_d + 8];
    pivot[j + 8][i + 8] = d[idx_d + V * 8 + 8];
    //row --> y fixed
    int idx_row_d = idx_y_col * V + idx_x;
    row[j][i] = d[idx_row_d];
    row[j + 8][i] = d[idx_row_d + V * 8];
    row[j][i + 8] = d[idx_row_d + 8];
    row[j + 8][i + 8] = d[idx_row_d + V * 8 + 8];
    //col --> x fixed
    int idx_col_d = idx_y * V + idx_x_row;
    col[j][i] = d[idx_col_d];
    col[j + 8][i] = d[idx_col_d + V * 8];
    col[j][i + 8] = d[idx_col_d + 8];
    col[j + 8][i + 8] = d[idx_col_d + V * 8 + 8];
    __syncthreads();
    #pragma unroll
    for(int k = 0; k < blocksize; k ++){
        pivot[j][i] = min(pivot[j][i], row[j][k] + col[k][i]);
        pivot[j + 8][i] = min(pivot[j + 8][i], row[j + 8][k] + col[k][i]);
        pivot[j][i + 8] = min(pivot[j][i + 8], row[j][k] + col[k][i + 8]);
        pivot[j + 8][i + 8] = min(pivot[j + 8][i + 8], row[j + 8][k] + col[k][i + 8]);
    }

    d[idx_d] = pivot[j][i];
    d[idx_d + V * 8] = pivot[j + 8][i];
    d[idx_d + 8] = pivot[j][i + 8];
    d[idx_d + V * 8 + 8] = pivot[j + 8][i + 8];
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

    // for GPU code
    size_t d_size = V_sq * sizeof(int);// use size_t instead of int !!!
    cudaHostRegister(d, d_size, cudaHostRegisterDefault);
    int* device_d;
    cudaMalloc(&device_d, d_size);
    cudaMemcpy(device_d, d, d_size, cudaMemcpyHostToDevice);
    dim3 blk(8, 8);
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
    handle_output(argv[2]);
    return 0;
}
