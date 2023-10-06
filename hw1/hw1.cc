#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <algorithm>
#define DEBUG_MODE 1
int main(int argc, char **argv)
{
  
    MPI_Init(&argc, &argv);
    int n = atoi(argv[1]);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *input_filename = argv[2];
    char *output_filename = argv[3];

    MPI_File input_file, output_file;
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
        start_offset = rank * ( n / size ) + ( n % size )
    }
    if(DEBUG_MODE){
        printf("rank = %d, data_to_solve = %d, start offset = %d", rank, data_to_solve, start_offset);
    }
    float* data = new float[data_to_solve];

    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * offset, data, 1, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);
    // sort local data first
    // [TODO] : optimize sorting algorithm
    float* first(&data[0]);
    float* last(first + data_to_solve);
    std::sort(first, last);
    

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * offset, data, 1, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}