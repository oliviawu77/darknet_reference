#include "layer.h"
#include <stdlib.h>

void free_sublayer(layer *l)
{
    if (l) {
        free_layer(*l);
        free(l);
    }
}

void free_layer(layer l)
{
    free_layer_custom(l, 0);
}

void free_layer_custom(layer l, int keep_cudnn_desc)
{

    //remove unused part
    if (l.mask)               free(l.mask);
    if (l.cweights)           free(l.cweights);
    if (l.indexes)            free(l.indexes);
    if (l.layers_delta)       free(l.layers_delta);
    if (l.map)                free(l.map);
    if (l.rand)               free(l.rand);
    if (l.cost)               free(l.cost);
    if (l.labels && !l.detection) free(l.labels);
    if (l.class_ids && !l.detection) free(l.class_ids);
    if (l.state)              free(l.state);
    if (l.prev_state)         free(l.prev_state);
    if (l.biases)             free(l.biases), l.biases = NULL;
    if (l.scales)             free(l.scales), l.scales = NULL;
    if (l.weights)            free(l.weights), l.weights = NULL;

    if (l.delta)              free(l.delta), l.delta = NULL;
    if (l.output)             free(l.output), l.output = NULL;
    if (l.activation_input)   free(l.activation_input), l.activation_input = NULL;
    if (l.squared)            free(l.squared);
    if (l.norms)              free(l.norms);
    if (l.mean)               free(l.mean), l.mean = NULL;
    if (l.variance)           free(l.variance), l.variance = NULL;
    if (l.mean_delta)         free(l.mean_delta), l.mean_delta = NULL;
    if (l.variance_delta)     free(l.variance_delta), l.variance_delta = NULL;
    if (l.rolling_mean)       free(l.rolling_mean), l.rolling_mean = NULL;
    if (l.rolling_variance)   free(l.rolling_variance), l.rolling_variance = NULL;
    if (l.x)                  free(l.x);
    if (l.x_norm)             free(l.x_norm);
    if (l.m)                  free(l.m);
    if (l.v)                  free(l.v);
    if (l.loss)               free(l.loss);
}
