#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer convolutional_layer;

#ifdef __cplusplus
extern "C" {
#endif

size_t get_convolutional_workspace_size(layer l);
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize);
void forward_convolutional_layer(const convolutional_layer layer, network_state state);

void add_bias(float *output, float *biases, int batch, int n, int size);

int convolutional_out_height(convolutional_layer layer);
int convolutional_out_width(convolutional_layer layer);

#ifdef __cplusplus
}
#endif

#endif
