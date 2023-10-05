#include <mpi.h>
#include <stdio.h>
int main(int argc, char *argv[]) {
  int rank, size;
  MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank); // the total number of process MPI_Comm_size(MPI_COMM_WORLD, &size); // the rank (id) of the calling process
  printf("Hello, World.  I am %d of %d\n", rank, size);
  MPI_Finalize();
return 0; }
