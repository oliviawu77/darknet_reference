#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

typedef layer avgpool_layer;

#ifdef __cplusplus
extern "C" {
#endif
image get_avgpool_image(avgpool_layer l);
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c);
void forward_avgpool_layer(const avgpool_layer l, network_state state);


#ifdef __cplusplus
}
#endif

#endif
