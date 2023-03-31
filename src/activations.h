#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "darknet.h"
#include "math.h"
#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif
ACTIVATION get_activation(char *s);

float activate(float x, ACTIVATION a);
void activate_array(float *x, const int n, const ACTIVATION a);

static inline float linear_activate(float x){return x;}
static inline float logistic_activate(float x){return 1.f/(1.f + expf(-x));}
static inline float leaky_activate(float x){return (x>0) ? x : .1f*x;}

#ifdef __cplusplus
}
#endif

#endif
