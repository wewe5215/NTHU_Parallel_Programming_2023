#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <algorithm>
#include <boost/sort/spreadsort/float_sort.hpp>
#define DEBUG_MODE 0
#define _0(x)	(x & 0xFF)
#define _1(x)	((x >> 8) & 0xFF)
#define _2(x)	((x >> 16) & 0xFF)
#define _3(x)	(x >> 24)
static inline unsigned int Float2Int(float input){
	unsigned int ret = *(unsigned int*)&input;
    ret ^= -(ret >> 31) | 0x80000000;
    return ret;
}

static inline float Int2Float(unsigned int input){
	input ^= ((input >> 31) - 1) | 0x80000000;
    return *(float*)&input;
}
//[TODO] :optimize radix sort
void radix_sort(float* data, int n)
{
	unsigned int* array = (unsigned int*)malloc(n * sizeof(unsigned int));
	unsigned int* sort = (unsigned int*)malloc(n * sizeof(unsigned int));

	// 4 histograms on the stack:
	const unsigned int kHist = 256;
	unsigned int b0[kHist * 4 + 5];
	unsigned int *b1 = b0 + kHist;
	unsigned int *b2 = b1 + kHist;
    unsigned int *b3 = b2 + kHist;

	memset(b0, 0, sizeof(unsigned int) * (kHist * 4 + 5));
	for(int i = 0; i < n; i++){
		array[i] = Float2Int(data[i]);
		b0[_0(array[i])]++;
		b1[_1(array[i])]++;
		b2[_2(array[i])]++;
        b3[_3(array[i])]++;
	}

    unsigned int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    unsigned int temp;
    for(int i = 0; i < kHist; i++){
        temp = b0[i] + sum0;
        b0[i] = sum0 - 1;
        sum0 = temp;

        temp = b1[i] + sum1;
        b1[i] = sum1 - 1;
        sum1 = temp;

        temp = b2[i] + sum2;
        b2[i] = sum2 - 1;
        sum2 = temp;

        temp = b3[i] + sum3;
        b3[i] = sum3 - 1;
        sum3 = temp;
    }

	for(int i = 0; i < n; i++){
        sort[++b0[_0(array[i])]] = array[i];
	}

	for(int i = 0; i < n; i++){
        array[++b1[_1(sort[i])]] = sort[i];
	}

	for(int i = 0; i < n; i++){
        sort[++b2[_2(array[i])]] = array[i];
	}

	for(int i = 0; i < n; i++){
        data[++b3[_3(sort[i])]] = Int2Float(sort[i]);
	}

	free(array);
	free(sort);
}

int main(int argc, char **argv)
{
    if (argc != 4) {
		fprintf(stderr, "must provide exactly 3 arguments!\n");
		return 1;
	}

    int rc;
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        printf ("Error starting MPI program. Terminating.\n"); 
        MPI_Abort (MPI_COMM_WORLD, rc);
    }

    int n = atoi(argv[1]);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *input_filename = argv[2];
    char *output_filename = argv[3];

    MPI_File input_file, output_file;

    MPI_Comm new_comm;

    // If size is greater than n, shrink the communicator
    if (n < size) {
        int color = (rank < n) ? 0 : MPI_UNDEFINED;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);
        if (color != 0) {
            MPI_Finalize();
            return 0; // Exit the processes not needed.
        }

        // Update rank and size for the new communicator
        MPI_Comm_rank(new_comm, &rank);
        MPI_Comm_size(new_comm, &size);
    } else {
        new_comm = MPI_COMM_WORLD;
    }
    int data_to_solve, start_offset;
    if(rank < n % size){
        data_to_solve = n / size + 1;
        if(rank == 0)start_offset = 0;
        else start_offset = rank * ( n / size ) + rank;
    }
    else{
        data_to_solve = n / size;
        start_offset = rank * ( n / size ) + ( n % size );
    }
    float* data = (float*) malloc((data_to_solve) * sizeof(float));
    float* number_buffer = (float*) malloc((n / size + 1) * sizeof(float));
    MPI_File_open(new_comm, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * start_offset, data, data_to_solve, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);
    unsigned int* array = (unsigned int*)malloc(data_to_solve * sizeof(unsigned int));
	unsigned int* sort = (unsigned int*)malloc(data_to_solve * sizeof(unsigned int));

	// 4 histograms on the stack:
	const unsigned int kHist = 256;
	unsigned int b0[kHist * 4 + 5];
	unsigned int *b1 = b0 + kHist;
	unsigned int *b2 = b1 + kHist;
    unsigned int *b3 = b2 + kHist;

	memset(b0, 0, sizeof(unsigned int) * (kHist * 4 + 5));
	for(int i = 0; i < data_to_solve; i++){
		array[i] = Float2Int(data[i]);
		b0[_0(array[i])]++;
		b1[_1(array[i])]++;
		b2[_2(array[i])]++;
        b3[_3(array[i])]++;
	}

    unsigned int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    unsigned int tsum;
    for(int i = 0; i < kHist; i++){
        tsum = b0[i] + sum0;
        b0[i] = sum0 - 1;
        sum0 = tsum;

        tsum = b1[i] + sum1;
        b1[i] = sum1 - 1;
        sum1 = tsum;

        tsum = b2[i] + sum2;
        b2[i] = sum2 - 1;
        sum2 = tsum;

        tsum = b3[i] + sum3;
        b3[i] = sum3 - 1;
        sum3 = tsum;
    }

	for(int i = 0; i < data_to_solve; i++){
        sort[++b0[_0(array[i])]] = array[i];
	}

	for(int i = 0; i < data_to_solve; i++){
        array[++b1[_1(sort[i])]] = sort[i];
	}

	for(int i = 0; i < data_to_solve; i++){
        sort[++b2[_2(array[i])]] = array[i];
	}

	for(int i = 0; i < data_to_solve; i++){
        data[++b3[_3(sort[i])]] = Int2Float(sort[i]);
	}

	free(array);
	free(sort);

    int round = size / 2 + 1;

    float* temp = (float*) malloc((data_to_solve) * sizeof(float));
    while(round --){
        if((rank % 2 == 0) && (rank != size - 1) && rank < n){
            int data_to_solve_next = n / size + (rank + 1 < n % size);
            MPI_Sendrecv(data + data_to_solve - 1, 1, MPI_FLOAT, rank + 1, 0,
                        number_buffer, 1, MPI_FLOAT, rank + 1, 0,
                        new_comm, MPI_STATUS_IGNORE
            );
            int i = 0, now = 0, neighbor = 0;
            if(number_buffer[0] < data[data_to_solve - 1]){
                MPI_Sendrecv(data, data_to_solve - 1, MPI_FLOAT, rank + 1, 0,
                            number_buffer + 1, data_to_solve_next - 1, MPI_FLOAT, rank + 1, 0,
                            new_comm, MPI_STATUS_IGNORE
                );
                for(i = 0; i < data_to_solve; i ++){
                    if(neighbor < data_to_solve_next){
                        if(number_buffer[neighbor] <= data[now]){
                            temp[i] = number_buffer[neighbor];
                            neighbor ++;
                        }
                        else{
                            temp[i] = data[now];
                            now ++;
                        }
                    }
                    else{
                        temp[i] = data[now];
                        now ++;
                    }
                }
                std::swap(temp, data);
            }

        }

        else if((rank % 2 == 1) && (rank != 0) && rank < n){
            int data_to_solve_prev = n / size + (rank - 1 < n % size);
            MPI_Sendrecv(data, 1, MPI_FLOAT, rank - 1, 0,
                        number_buffer + data_to_solve_prev - 1, 1, MPI_FLOAT, rank - 1, 0,
                        new_comm, MPI_STATUS_IGNORE
            );
            int i = 0, now = data_to_solve - 1, neighbor = data_to_solve_prev - 1;
            if(number_buffer[data_to_solve_prev - 1] > data[0]){
                MPI_Sendrecv(data + 1, data_to_solve - 1, MPI_FLOAT, rank - 1, 0,
                        number_buffer, data_to_solve_prev - 1, MPI_FLOAT, rank - 1, 0,
                        new_comm, MPI_STATUS_IGNORE
                );
                for(i = data_to_solve - 1 ; i >= 0; i --){
                    if(number_buffer[neighbor] >= data[now] && neighbor >= 0){
                        temp[i] = number_buffer[neighbor];
                        neighbor --;
                    }
                    else{
                        temp[i] = data[now];
                        now --;
                    }
                }
                std::swap(temp, data);
            }
        }

        if((rank % 2 == 0) && (rank != 0) && rank < n){
            int data_to_solve_prev = n / size + (rank - 1 < n % size);
            MPI_Sendrecv(data, 1, MPI_FLOAT, rank - 1, 0,
                        number_buffer + data_to_solve_prev - 1, 1, MPI_FLOAT, rank - 1, 0,
                        new_comm, MPI_STATUS_IGNORE
            );
            int i = 0, now = data_to_solve - 1, neighbor = data_to_solve_prev - 1;
            if(number_buffer[data_to_solve_prev - 1] > data[0]){
                MPI_Sendrecv(data + 1, data_to_solve - 1, MPI_FLOAT, rank - 1, 0,
                            number_buffer, data_to_solve_prev - 1, MPI_FLOAT, rank - 1, 0,
                            new_comm, MPI_STATUS_IGNORE
                );
                for(i = data_to_solve - 1 ; i >= 0; i --){
                    if(number_buffer[neighbor] >= data[now] && neighbor >= 0){
                        temp[i] = number_buffer[neighbor];
                        neighbor --;
                    }
                    else{
                        temp[i] = data[now];
                        now --;
                    }
                }
                std::swap(temp, data);
            }
        }
        else if((rank % 2 == 1) && (rank != size - 1) && rank < n){
            int data_to_solve_next = n / size + (rank + 1 < n % size);
            MPI_Sendrecv(data + data_to_solve - 1, 1, MPI_FLOAT, rank + 1, 0,
                        number_buffer, 1, MPI_FLOAT, rank + 1, 0,
                        new_comm, MPI_STATUS_IGNORE
            );
            int i = 0, now = 0, neighbor = 0;
            if(number_buffer[0] < data[data_to_solve - 1]){
                MPI_Sendrecv(data, data_to_solve - 1, MPI_FLOAT, rank + 1, 0,
                            number_buffer + 1, data_to_solve_next - 1, MPI_FLOAT, rank + 1, 0,
                            new_comm, MPI_STATUS_IGNORE
                );
                for(i = 0; i < data_to_solve; i ++){
                    if(neighbor < data_to_solve_next){
                        if(number_buffer[neighbor] <= data[now]){
                            temp[i] = number_buffer[neighbor];
                            neighbor ++;
                        }
                        else{
                            temp[i] = data[now];
                            now ++;
                        }
                    }
                    else{
                        temp[i] = data[now];
                        now ++;
                    }
                }
                std::swap(temp, data);
            }
        }
    }
    free(number_buffer);
    free(temp);
    MPI_File_open(new_comm, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * start_offset, data, data_to_solve, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);
    MPI_Finalize();
    return 0;
}