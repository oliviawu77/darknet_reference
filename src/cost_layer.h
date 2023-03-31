#ifndef COST_LAYER_H
#define COST_LAYER_H
#include "layer.h"
#include "network.h"

typedef layer cost_layer;

#ifdef __cplusplus
extern "C" {
#endif
COST_TYPE get_cost_type(char *s);
cost_layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type);
void forward_cost_layer(const cost_layer l, network_state state);

#ifdef __cplusplus
}
#endif
#endif
