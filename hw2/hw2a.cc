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
#include <pthread.h>
#include <math.h>
#include <emmintrin.h>
const char* filename;
int iters, width, height, ncpus;
double left, right, lower, upper;
double XCoeff, YCoeff;
cpu_set_t cpu_set;
size_t row_size;
png_bytep row;
png_bytep* rows;
pthread_mutex_t mutex;
int global_j = 0;
static inline void write_color(int p, png_bytep color){
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
void write_png(const char* filename, int iters, int width, int height) {
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
        png_write_row(png_ptr, rows[y]);
        free(rows[y]);
    }
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void* mandelbrot(void* threadid){
    int j = 0;
    /* mandelbrot set */
    double XCoeff, YCoeff;
    XCoeff = ((right - left) / width);
    YCoeff = ((upper - lower) / height);
    while(true) {
        pthread_mutex_lock(&mutex);
        if(global_j == height){
            pthread_mutex_unlock(&mutex);
            break;
        }
        else{
            j = global_j;
            global_j++;
        }
        pthread_mutex_unlock(&mutex);
        __m128d x, y, x_sq, y_sq, x0, y0, num_1, num_2, length_squared, repeats;
        num_1 = _mm_set_pd((double)1.0, (double)1.0);
        num_2 = _mm_set_pd((double)2.0, (double)2.0);
        rows[(height - 1 - j)] = (png_bytep)malloc(row_size);
        memset(rows[(height - 1 - j)], 0, row_size);
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
                    write_color(repeats[0], rows[(height - 1 - j)] + i * 3);
                    write_color(repeats[1], rows[(height - 1 - j)] + (i+1) * 3);
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
            write_color(repeats, rows[(height - 1 - j)] + (width-1) * 3);
        }
    }
    pthread_exit(NULL);
}
int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    ncpus = CPU_COUNT(&cpu_set);

    /* argument parsing */
    assert(argc == 9);
    filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    rows = (png_bytep*)malloc(height * sizeof(png_bytep));
    row_size = 3 * width * sizeof(png_byte);
    XCoeff = ((right - left) / width);
    YCoeff = ((upper - lower) / height);
    pthread_t threads[ncpus];
    int rc;
    int ID[ncpus];
    int t;
    global_j = 0;
    pthread_mutex_init (&mutex, NULL);
    for (t = 0; t < ncpus; t++) {
        ID[t] = t;
        rc = pthread_create(&threads[t], NULL, mandelbrot, (void*)&ID[t]);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }
	for(t = 0; t < ncpus; t++)
		pthread_join(threads[t], NULL);
    /* draw and cleanup */
    // mandelbrot();
    pthread_mutex_destroy(&mutex);
    write_png(filename, iters, width, height);

}