#include "convolutional_layer.h"
#include "utils.h"
#include "im2col.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

#ifdef __cplusplus
#define PUT_IN_REGISTER
#else
#define PUT_IN_REGISTER register
#endif

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

size_t get_workspace_size32(layer l){

    return (size_t)l.out_h*l.out_w*l.size*l.size*(l.c / l.groups)*sizeof(float);
}

size_t get_workspace_size16(layer l) {

    return 0;

}

size_t get_convolutional_workspace_size(layer l) {
    size_t workspace_size = get_workspace_size32(l);
    size_t workspace_size16 = get_workspace_size16(l);
    if (workspace_size16 > workspace_size) workspace_size = workspace_size16;
    return workspace_size;
}

convolutional_layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride, int dilation, int padding, ACTIVATION activation, int batch_normalize, int adam, int index, int train)
{
    int total_batch = batch*steps;
    int i;
    convolutional_layer l = { (LAYER_TYPE)0 };
    l.type = CONVOLUTIONAL;
    l.train = train;

    if (groups < 1) groups = 1;

    l.index = index;
    l.h = h;
    l.w = w;
    l.c = c;
    l.groups = groups;
    l.n = n;
    l.batch = batch;
    l.steps = steps;
    l.stride = stride;
    l.dilation = dilation;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;
    l.nweights = (c / groups) * n * size * size;


    l.weights = (float*)xcalloc(l.nweights, sizeof(float));
    l.biases = (float*)xcalloc(n, sizeof(float));

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c/groups));

    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;
    l.activation = activation;

    l.output = (float*)xcalloc(total_batch*l.outputs, sizeof(float));

    l.forward = forward_convolutional_layer;
    //remove backward layer and update layer
    //remove binary and xnor condition

    if(batch_normalize){
        l.scales = (float*)xcalloc(n, sizeof(float));
        for (i = 0; i < n; ++i) {
            l.scales[i] = 1;
        }
        //remove unused part
        l.rolling_mean = (float*)xcalloc(n, sizeof(float));
        l.rolling_variance = (float*)xcalloc(n, sizeof(float));
    }


    //remove unused part
    l.workspace_size = get_convolutional_workspace_size(l);

    //fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    l.bflops = (2.0 * l.nweights * l.out_h*l.out_w) / 1000000000.;

    fprintf(stderr, "conv  ");

    if (groups > 1) fprintf(stderr, "%5d/%4d ", n, groups);
    else           fprintf(stderr, "%5d      ", n);

    fprintf(stderr, "%2d x%2d/%2d   ", size, size, stride);
    
    fprintf(stderr, "%4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", w, h, c, l.out_w, l.out_h, l.out_c, l.bflops);
    //remove unused part
    return l;
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

//copy from https://github.com/AlexeyAB/darknet/blob/bc810016a1f2eadf33a1ac800b64962a42f3d402/src/convolutional_layer.c
void forward_convolutional_layer(convolutional_layer l, network_state state)
{
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    int i;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    //remove xnor

    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;


    float *a = l.weights;
    float *b = state.workspace;
    float *c = l.output;

    for(i = 0; i < l.batch; ++i){
        im2col_cpu(state.input, l.c, l.h, l.w, 
                l.size, l.stride, l.pad, b);
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        c += n*m;
        state.input += l.c*l.h*l.w;
    }

    //remove batch normalize here
    add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);

    activate_array(l.output, m*n*l.batch, l.activation);
}

