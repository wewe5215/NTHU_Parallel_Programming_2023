#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // the total number of process MPI_Comm_size(MPI_COMM_WORLD, &size); // the rank (id) of the calling process
  MPI_Comm_size(MPI_COMM_WORLD, &size);

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
  unsigned long long pixels_local = 0;
  unsigned long long pixels_global = 0;

  // Parallelize the loop
  for (unsigned long long x = rank; x < r; x = x + size) {
      unsigned long long y = ceil(sqrtl(r * r - x * x));
      pixels_local += y;
      pixels_local %= k;
  }

  // Reduce the local results to a global result
  MPI_Reduce(&pixels_local, &pixels_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if(rank == 0)
	  printf("%llu\n", (4 * pixels_global) % k);
  MPI_Finalize();
  return 0;
}