#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <emmintrin.h>
#define n height
static inline void write_color(int p, png_bytep color, int iters){
    if (p != iters) {
        if (p & 16) {
            color[0] = 240;
            color[1] = color[2] = p % 16 * 16;
        } else {
            color[0] = p % 16 * 16;
        }
    }
    return;
}
inline void write_png(const char* filename, int iters, int width, int height, png_bytep* rows, int* mapping) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    for (int y = 0; y < height; ++y) {
        png_write_row(png_ptr, rows[mapping[y]]);
        // free(rows[y]);
    }
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

inline void mandelbrot(double left, double right, int width, double upper, double lower, int height, \
                       int rank, int start_offset, int data_to_solve, size_t row_size, \
                       int iters, png_bytep* rows, int ncpus, int size){

    /* mandelbrot set */
    double XCoeff, YCoeff;
    XCoeff = ((right - left) / width);
    YCoeff = ((upper - lower) / height);
    #pragma omp parallel num_threads(ncpus)
	{
		#pragma omp for schedule(dynamic)
            for (int j = rank; j < height; j += size) {
                __m128d x, y, x_sq, y_sq, x0, y0, num_1, num_2, length_squared, repeats;
                num_1 = _mm_set_pd((double)1.0, (double)1.0);
                num_2 = _mm_set_pd((double)2.0, (double)2.0);
                int idx = j / size;
                memset(rows[idx], 0, row_size);
                for (int i = 0; i < width; i += 2) {
                    if(i + 1 >= width)break;
                    x0[0] = i * XCoeff + left;
                    x0[1] = (i+1) * XCoeff + left;
                    y0 = _mm_set_pd((double)j * YCoeff + lower, (double)j * YCoeff + lower);
                    x = _mm_set_pd((double)0.0, (double)0.0);
                    y = x_sq = y_sq = length_squared = repeats = x;
                    while(true){
                        if (repeats[0] < iters && length_squared[0] < 4 \
                            && repeats[1] < iters && length_squared[1] < 4) {
                            y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(num_2, x), y), y0);
                            x = _mm_add_pd(_mm_sub_pd(x_sq, y_sq), x0);
                            x_sq = _mm_mul_pd(x, x);
                            y_sq =_mm_mul_pd(y, y);
                            length_squared = _mm_add_pd(x_sq, y_sq);
                            repeats = _mm_add_pd(num_1, repeats);
                        }
                        else if(repeats[1] < iters && length_squared[1] < 4){
                            y[1] = 2 * x[1] * y[1] + y0[1];
                            x[1] = x_sq[1] - y_sq[1] + x0[1];
                            x_sq[1] = x[1] * x[1];
                            y_sq[1] = y[1] * y[1];
                            length_squared[1] = x_sq[1] + y_sq[1];
                            ++repeats[1];
                        }
                        else if((repeats[0] < iters && length_squared[0] < 4)){
                            y[0] = 2 * x[0] * y[0] + y0[0];
                            x[0] = x_sq[0] - y_sq[0] + x0[0];
                            x_sq[0] = x[0] * x[0];
                            y_sq[0] = y[0] * y[0];
                            length_squared[0] = x_sq[0] + y_sq[0];
                            ++repeats[0];
                        }
                        else{
                            write_color(repeats[0], rows[idx] + i * 3, iters);
                            write_color(repeats[1], rows[idx] + (i+1) * 3, iters);
                            break;
                        }
                    }
                }
                if(width % 2){
                    double x0 = (width-1) * XCoeff + left;

                    int repeats = 0;
                    double x = 0, y = 0;
                    double x_sq = 0, y_sq = 0;
                    double length_squared = 0;
                    while (repeats < iters && length_squared < 4) {
                        y = 2 * x * y + y0[0];
                        x = x_sq - y_sq + x0;
                        x_sq = x * x;
                        y_sq = y * y;
                        length_squared = x_sq + y_sq;
                        ++repeats;
                    }
                    // printf("rank = %d\n", rank);
                    write_color(repeats, rows[idx] + (width-1) * 3, iters);
                    // printf("rank = %d\n", rank);
                }
            }
    }
    return;
}
int main(int argc, char** argv) {
    
    double XCoeff, YCoeff;
    cpu_set_t cpu_set;
    png_bytep row;
    int rank, size;
    int data_to_solve, start_offset;
    /* detect how many CPUs are available */
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);
    size_t row_size = 3 * width * sizeof(png_byte);
    XCoeff = ((right - left) / width);
    YCoeff = ((upper - lower) / height);
    int rc;
    rc = MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if(rank < height % size){
        data_to_solve = height / size + 1;
        if(rank == 0)start_offset = 0;
        else start_offset = rank * ( height / size ) + rank;
    }
    else{
        data_to_solve = height / size;
        start_offset = rank * ( height / size ) + ( height % size );
    }
    png_bytep local_rows = (png_bytep)malloc(data_to_solve * row_size);
    png_bytep global_rows = (png_bytep)malloc(height * row_size);
    png_bytep* local_rows_ptr = (png_bytep*)malloc(data_to_solve * sizeof(png_bytep));
    for(int i = 0; i < data_to_solve; ++i){
        local_rows_ptr[i] = &local_rows[3 * width * i];
    }
    mandelbrot( left, right, width, upper, lower, height, \
                rank, start_offset, data_to_solve, row_size, \
                iters, local_rows_ptr, ncpus, size);
    int* recvcounts;
    int* displs;
    // recvcounts, displs, recvtype: (significant only at root)
    if(rank == 0){
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        for(int i = 0; i < size; i ++){
            recvcounts[i] = n / size;
            if(i < n % size)recvcounts[i] += 1;
            recvcounts[i] = recvcounts[i] * (3 * width);
            if(i == 0){
                displs[i] = 0;
            }
            else{
                displs[i] = displs[i - 1] + recvcounts[i - 1];
            }
        }
    }
    MPI_Gatherv(local_rows, data_to_solve * width * 3, MPI_UNSIGNED_CHAR, \
                global_rows, recvcounts, displs, \
                MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    if(rank == 0){
        int* mapping = (int*)malloc(height * sizeof(int));
        int j, k = 0;
		for (int i = 0; i < size; i++) {
			for (j = i; j < height; j += size) {
				mapping[height - j - 1] = k++;
			}
		}
        png_bytep* global_rows_ptr = (png_bytep*)malloc(height * sizeof(png_bytep));
        for(int i = 0; i < height; ++i){
            global_rows_ptr[i] = &global_rows[3 * width * i];
        }
        write_png(filename, iters, width, height, global_rows_ptr, mapping);
    }
    free(global_rows);
    free(local_rows);
    MPI_Finalize();
}