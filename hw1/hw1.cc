#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <algorithm>
#include <boost/sort/spreadsort/float_sort.hpp>
#define DEBUG_MODE 0
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
    // 1 ≤ n ≤ 536870911, n is the number of data
    // The input file contains n 32-bit floats in binary format
    /*
        to decide the start offset to read the file
        if n > size, then it means that each process have to solve more than 1 data
        e.g. n = 6, size = 4, then the process with rank 0, 1 have to solve 2 data
    */ 
    int data_to_solve, start_offset;
    //[TODO] : what if n < size ? which means that we don't need that much processes
    //how about removing the ones that isn't necessary?
    if(rank < n % size){
        data_to_solve = n / size + 1;
        //rank = 0 -> 0, 1;rank = 1 -> 2, 3;rank = 2 -> 4
        if(rank == 0)start_offset = 0;
        else start_offset = rank * ( n / size ) + rank;
    }
    else{
        data_to_solve = n / size;
        start_offset = rank * ( n / size ) + ( n % size );
    }
    if(DEBUG_MODE){
        printf("rank = %d, data_to_solve = %d, start offset = %d\n", rank, data_to_solve, start_offset);
    }
    float* data = new float[data_to_solve];
    float* number_buffer_prev = new float[n / size + 1];
    float* number_buffer_next = new float[n / size + 1];
    MPI_File_open(new_comm, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * start_offset, data, data_to_solve, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);
    // sort local data first
    // [TODO] : optimize sorting algorithm
    float* first(&data[0]);
    float* last(first + data_to_solve);
    std::sort(first, last);
    // boost::sort::spreadsort::float_sort(first, last);
    int round = size / 2 + 1;
    // odd even sort
    /*  [TODO] :
        since every local data are sorted, we can send the marginal data first
        and decide to swap the content or not
    */
   if(DEBUG_MODE){
        // if(data_to_solve == 1){
        //     printf("here is rank %d initial value\ndata[0] = %f\n\n", rank,  data[0]); 
        // }
        // else if (data_to_solve == 2){
        //     printf("here is rank %d initial value\ndata[0] = %f, data[1] = %f\n\n", rank,  data[0], data[1]); 
        // }
        // else if(data_to_solve == 3)
        //    printf("here is rank %d initial value\ndata[0] = %f, data[1] = %f, data[2] = %f\n\n", rank,  data[0], data[1], data[2]);
        printf("this is rank %d initial value\n", rank);
        for(int i = 0; i < data_to_solve; i ++){
            printf("data[%d] = %f\n", i, data[i]);
        }
   
   }
    bool swap = false;
    while(round --){
        if((rank % 2 == 0) && (rank != size - 1) && rank < n){
            /*
                For each process with rank P-1, 
                compare its number with the number sent by the process with rank P and 
                send the larger one back to the process with rank P
            */
            int data_to_solve_next = n / size + (rank + 1 < n % size);
            MPI_Sendrecv(data + data_to_solve - 1, 1, MPI_FLOAT, rank + 1, 0,
                        number_buffer_next, 1, MPI_FLOAT, rank + 1, 0,
                        new_comm, MPI_STATUS_IGNORE
            );
            int i = 0, now = 0, neighbor = 0;
            float* temp = new float[data_to_solve];
            if(DEBUG_MODE){
                printf("number_buffer_next[0] = %f, data[data_to_solve - 1] = %f\n", number_buffer_next[0], data[data_to_solve - 1]);
            }
            if(number_buffer_next[0] < data[data_to_solve - 1]){
                MPI_Sendrecv(data, data_to_solve - 1, MPI_FLOAT, rank + 1, 0,
                            number_buffer_next + 1, data_to_solve_next - 1, MPI_FLOAT, rank + 1, 0,
                            new_comm, MPI_STATUS_IGNORE
                );
                if(DEBUG_MODE)printf("swapping 1!\n");
                for(i = 0; i < data_to_solve; i ++){
                    if(neighbor < data_to_solve_next){
                        if(number_buffer_next[neighbor] <= data[now]){
                            temp[i] = number_buffer_next[neighbor];
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
                swap = true;
                if(DEBUG_MODE){
                    printf("here is rank %d\n", rank);
                    for(int i = 0; i < data_to_solve; i ++){
                        printf("data[%d] = %f\n", i, data[i]);
                    }
                }
            }
            
            

        }

        else if((rank % 2 == 1) && (rank != 0) && rank < n){
            /*For each process with odd rank P, send its number to the process with rank P-1*/
            int data_to_solve_prev = n / size + (rank - 1 < n % size);
            MPI_Sendrecv(data, 1, MPI_FLOAT, rank - 1, 0,
                        number_buffer_prev + data_to_solve_prev - 1, 1, MPI_FLOAT, rank - 1, 0,
                        new_comm, MPI_STATUS_IGNORE
            );
            int i = 0, now = data_to_solve - 1, neighbor = data_to_solve_prev - 1;
            float* temp = new float[data_to_solve];
            if(DEBUG_MODE){
                printf("number_buffer_prev[data_to_solve_prev] = %f, data[0] = %f\n", number_buffer_prev[data_to_solve_prev - 1], data[0]);
            }
            if(number_buffer_prev[data_to_solve_prev - 1] > data[0]){
                if(DEBUG_MODE)printf("swapping 2!\n");
                MPI_Sendrecv(data + 1, data_to_solve - 1, MPI_FLOAT, rank - 1, 0,
                        number_buffer_prev, data_to_solve_prev - 1, MPI_FLOAT, rank - 1, 0,
                        new_comm, MPI_STATUS_IGNORE
                );
                for(i = data_to_solve - 1 ; i >= 0; i --){
                    if(number_buffer_prev[neighbor] >= data[now] && neighbor >= 0){
                        if(DEBUG_MODE){
                            printf("number_buffer_prev[neighbor] = %f, data[now] = %f\n", number_buffer_prev[neighbor], data[now]);
                        }
                        temp[i] = number_buffer_prev[neighbor];
                        neighbor --;
                    }
                    else{
                        temp[i] = data[now];
                        now --;
                    }
                }
                
                
                std::swap(temp, data);
                swap = true;
                if(DEBUG_MODE){
                    printf("here is rank %d\n", rank);
                    for(int i = 0; i < data_to_solve; i ++){
                        printf("data[%d] = %f\n", i, data[i]);
                    }
                }
            }
            
            
        }

        if((rank % 2 == 0) && (rank != 0) && rank < n){
            /*For each process with even rank Q, send its number to the process with rank Q-1*/
            int data_to_solve_prev = n / size + (rank - 1 < n % size);
            MPI_Sendrecv(data, 1, MPI_FLOAT, rank - 1, 0,
                        number_buffer_prev + data_to_solve_prev - 1, 1, MPI_FLOAT, rank - 1, 0,
                        new_comm, MPI_STATUS_IGNORE
            );
            int i = 0, now = data_to_solve - 1, neighbor = data_to_solve_prev - 1;
            float* temp = new float[data_to_solve];
            if(DEBUG_MODE){
                printf("number_buffer_prev[data_to_solve_prev] = %f, data[0] = %f\n", number_buffer_prev[data_to_solve_prev - 1], data[0]);
            }
            if(number_buffer_prev[data_to_solve_prev - 1] > data[0]){
                if(DEBUG_MODE)printf("swapping 3!\n");
                MPI_Sendrecv(data + 1, data_to_solve - 1, MPI_FLOAT, rank - 1, 0,
                            number_buffer_prev, data_to_solve_prev - 1, MPI_FLOAT, rank - 1, 0,
                            new_comm, MPI_STATUS_IGNORE
                );
                for(i = data_to_solve - 1 ; i >= 0; i --){
                    if(number_buffer_prev[neighbor] >= data[now] && neighbor >= 0){
                        if(DEBUG_MODE){
                            printf("number_buffer_prev[neighbor] = %f, data[now] = %f\n", number_buffer_prev[neighbor], data[now]);
                        }
                        temp[i] = number_buffer_prev[neighbor];
                        neighbor --;
                    }
                    else{
                        temp[i] = data[now];
                        now --;
                    }
                }
                std::swap(temp, data);
                swap = true;
                if(DEBUG_MODE){
                    printf("here is rank %d\n", rank);
                    for(int i = 0; i < data_to_solve; i ++){
                        printf("data[%d] = %f\n", i, data[i]);
                    }
                }
            }
            
        }
        else if((rank % 2 == 1) && (rank != size - 1) && rank < n){
            /*
                For each process with rank Q-1, 
                compare its number with the number sent by the process with rank Q and 
                send the larger one back to the process with rank Q
            */
            int data_to_solve_next = n / size + (rank + 1 < n % size);
            MPI_Sendrecv(data + data_to_solve - 1, 1, MPI_FLOAT, rank + 1, 0,
                        number_buffer_next, 1, MPI_FLOAT, rank + 1, 0,
                        new_comm, MPI_STATUS_IGNORE
            );
            int i = 0, now = 0, neighbor = 0;
            float* temp = new float[data_to_solve];
            if(DEBUG_MODE){
                printf("number_buffer_next[0] = %f, data[data_to_solve - 1] = %f\n", number_buffer_next[0], data[data_to_solve - 1]);
            }
            if(number_buffer_next[0] < data[data_to_solve - 1]){
                MPI_Sendrecv(data, data_to_solve - 1, MPI_FLOAT, rank + 1, 0,
                            number_buffer_next + 1, data_to_solve_next - 1, MPI_FLOAT, rank + 1, 0,
                            new_comm, MPI_STATUS_IGNORE
                );
                if(DEBUG_MODE)printf("swapping 4!\n");
                for(i = 0; i < data_to_solve; i ++){
                    if(neighbor < data_to_solve_next){
                        if(number_buffer_next[neighbor] <= data[now]){
                            temp[i] = number_buffer_next[neighbor];
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
                swap = true;
                if(DEBUG_MODE){
                    printf("here is rank %d\n", rank);
                    for(int i = 0; i < data_to_solve; i ++){
                        printf("data[%d] = %f\n", i, data[i]);
                    }
                }
            }
            
        }
    }
    delete[] number_buffer_prev;
    delete[] number_buffer_next;
    MPI_File_open(new_comm, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * start_offset, data, data_to_solve, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    if(DEBUG_MODE){
        printf("here is rank %d\n", rank);
        for(int i = 0; i < data_to_solve; i ++){
            printf("data[%d] = %f\n", i, data[i]);
        }
    }
    MPI_Finalize();
    return 0;
}