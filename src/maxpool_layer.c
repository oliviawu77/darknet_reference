#include "maxpool_layer.h"
#include "convolutional_layer.h"
#include "utils.h"
#include "gemm.h"
#include <stdio.h>

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding, int avgpool)
{
    maxpool_layer l = { (LAYER_TYPE)0 };
    l.type = MAXPOOL;

    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;

    l.out_w = (w + padding - size) / stride + 1;
    l.out_h = (h + padding - size) / stride + 1;
    l.out_c = c;
    
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;
    int output_size = l.out_h * l.out_w * l.out_c * batch;

    l.output = (float*)xcalloc(output_size, sizeof(float));

    l.forward = forward_maxpool_layer;

	l.bflops = (l.size*l.size*l.c * l.out_h*l.out_w) / 1000000000.;
    
    fprintf(stderr, "max               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);

    return l;
}


void forward_maxpool_layer(const maxpool_layer l, network_state state)
{

    forward_maxpool_layer_avx(state.input, l.output, l.indexes, l.size, l.w, l.h, l.out_w, l.out_h, l.c, l.pad, l.stride, l.batch);
    
}
