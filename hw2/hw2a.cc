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
const char* filename;
int iters, width, height, ncpus;
double left, right, lower, upper;
cpu_set_t cpu_set;
size_t row_size;
png_bytep row;
png_bytep* rows;
int* image;
void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
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
    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    rows = (png_bytep*)malloc(height * sizeof(png_bytep));
    row_size = 3 * width * sizeof(png_byte);
    assert(image);

    /* mandelbrot set */
    double XCoeff, YCoeff;
    XCoeff = ((right - left) / width);
    YCoeff = ((upper - lower) / height);
    for (int j = 0; j < height; ++j) {
        double y0 = j * YCoeff + lower;
        rows[(height - 1 - j)] = (png_bytep)malloc(row_size);
        memset(rows[(height - 1 - j)], 0, row_size);
        for (int i = 0; i < width; ++i) {
            double x0 = i * XCoeff + left;

            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4) {
                double x_sq = x * x, y_sq = y * y;
                double temp = x_sq - y_sq + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x_sq + y_sq;
                ++repeats;
            }
            image[j * width + i] = repeats;
            int p = image[j * width + i];
            png_bytep color = rows[(height - 1 - j)] + i * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
    }
    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
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

}

