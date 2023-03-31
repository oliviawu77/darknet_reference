#ifndef IM2COL_H
#define IM2COL_H

#include <stddef.h>
#include <stdint.h>
#include "darknet.h"

#ifdef __cplusplus
extern "C" {
#endif
void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);
float im2col_get_pixel(float* im, int height, int width, int channels,
    int row, int col, int channel, int pad);

#ifdef __cplusplus
}
#endif
#endif
