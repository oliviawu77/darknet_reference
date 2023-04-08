// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H
#include "darknet.h"

#include <stdint.h>
#include "layer.h"


#include "image.h"

#ifdef __cplusplus
extern "C" {
#endif



char *get_layer_string(LAYER_TYPE a);

network make_network(int n);
void forward_network(network net, network_state state);

float *get_network_output(network net);
float *get_network_output_layer(network net, int i);
int get_network_output_size(network net);
void set_batch_network(network *net, int b);


#ifdef __cplusplus
}
#endif

#endif
