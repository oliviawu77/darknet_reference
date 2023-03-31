#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "leaky")==0) return LEAKY;
}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case LOGISTIC:
            return logistic_activate(x);
        case LEAKY:
            return leaky_activate(x);
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    if (a == LINEAR) {}
    else if (a == LEAKY) {
        for (i = 0; i < n; ++i) {
            x[i] = leaky_activate(x[i]);
        }
    }
    else if (a == LOGISTIC) {
        for (i = 0; i < n; ++i) {
            x[i] = logistic_activate(x[i]);
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            x[i] = activate(x[i], a);
        }
    }
}

