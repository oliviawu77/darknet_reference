#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "activations.h"
#include "assert.h"
#include "avgpool_layer.h"
#include "blas.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "list.h"
#include "maxpool_layer.h"
#include "option_list.h"
#include "parser.h"
#include "softmax_layer.h"
#include "utils.h"

typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    int train;
    network net;
} size_params;

convolutional_layer parse_convolutional(int batch_normalize, int filter, int size, int stride, int pad, char* activation_s, size_params params)
{
    int n = filter;
    int groups = 1;

    int dilation = 1;
    if (size == 1) dilation = 1;
    int padding;
    if(pad) padding = size/2;

    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;

    if(!(h && w && c)) error("Layer before convolutional layer must output image.", DARKNET_LOC);

    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize);
    //fprintf(stderr, "batch: %d, h: %d, w: %d, c: %d, n: %d, groups: %d, size: %d, stride: %d, padding: %d, activation: %s, batch_normalize: %d, \n", batch, h, w, c, n, groups, size, stride, padding, activation, batch_normalize);
    //printf("batch: %d, h: %d, w: %d, c: %d, n: %d, groups: %d, size: %d, stride: %s, padding: %d, activation: %s, batch_normalize: %d, \n", batch, h, w, c, n, groups, size, stride, padding, activation, batch_normalize);

    return layer;
}

softmax_layer parse_softmax(size_params params)
{
	int groups = 1;
	softmax_layer layer = make_softmax_layer(params.batch, params.inputs, groups);

	layer.w = params.w;
	layer.h = params.h;
	layer.c = params.c;
	return layer;
}

cost_layer parse_cost(size_params params)
{
    char *type_s = "sse";
    COST_TYPE type = get_cost_type(type_s);

    cost_layer layer = make_cost_layer(params.batch, params.inputs, type);

    return layer;
}


maxpool_layer parse_maxpool(size_params params)
{
    int stride = 2;
    int size = 2;
    int padding = 1;
    const int avgpool = 0;

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;

    maxpool_layer layer = make_maxpool_layer(batch, h, w, c, size, stride, padding, avgpool);
    return layer;
}

avgpool_layer parse_avgpool(size_params params)
{
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;

    avgpool_layer layer = make_avgpool_layer(batch,w,h,c);
    return layer;
}

learning_rate_policy get_policy(char *s)
{
	return POLY;
}

void parse_net_options(network *net)
{
    net->max_batches = 1600000;
    net->batch = 1;
    int subdivs = 1;
    net->time_steps = 1;
    net->batch /= subdivs;          // mini_batch
    net->batch *= net->time_steps;  // mini_batch * time_steps
    net->subdivisions = subdivs;    // number of mini_batches

    *net->seen = 0;
    *net->cur_iteration = 0;
    net->workspace_size_limit = (size_t)1024*1024 * 1024;  // 1024 MB by default

    net->h = 224;
    net->w = 224;
    net->c = 3;
    net->inputs = net->h * net->w * net->c;

    if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied", DARKNET_LOC);

    char *policy_s = "poly";
    net->policy = get_policy(policy_s);

}

network parse_network_cfg_custom(int batch, int time_steps)
{
    network net = make_network(23);
    size_params params;

    parse_net_options(&net);
    
    params.h = net.h;
    params.w = net.w;
    params.c = net.c;
    params.inputs = net.inputs;
    if (batch > 0) net.batch = batch;
    if (time_steps > 0) net.time_steps = time_steps;
    if (net.batch < 1) net.batch = 1;
    if (net.time_steps < 1) net.time_steps = 1;
    if (net.batch < net.time_steps) net.batch = net.time_steps;
    params.batch = net.batch;
    params.time_steps = net.time_steps;
    params.net = net;
    printf("mini_batch = %d, batch = %d, time_steps = %d \n", net.batch, net.batch * net.subdivisions, net.time_steps);

    int avg_outputs = 0;
    int avg_counter = 0;
    float bflops = 0;
    size_t workspace_size = 0;
    size_t max_inputs = 0;
    size_t max_outputs = 0;

    int count = 0;

    fprintf(stderr, "   layer   filters  size/strd(dil)      input                output\n");
    
    int batch_normalize, filter, size, stride, pad;
    char* activation_s;
    activation_s = (char*)xmalloc(sizeof(char)*512);

    while(count < 23){
        params.index = count;
        fprintf(stderr, "%4d ", count);
        layer l = { (LAYER_TYPE)0 };
        switch(count){
            case 0:
                batch_normalize = 1;
                filter = 16;
                size = 3;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "leaky");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);
            break;
            case 1:
                l = parse_maxpool(params);
            break;
            case 2:
                batch_normalize = 1;
                filter = 32;
                size = 3;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "leaky");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);
            break;
            case 3:
                l = parse_maxpool(params);
            break;
            case 4:
                batch_normalize = 1;
                filter = 16;
                size = 1;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "leaky");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);
            break;
            case 5:
                batch_normalize = 1;
                filter = 128;
                size = 3;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "leaky");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);
            break;
            case 6:
                batch_normalize = 1;
                filter = 16;
                size = 1;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "leaky");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);
            break;
            case 7:
                batch_normalize = 1;
                filter = 128;
                size = 3;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "leaky");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);
            break;
            case 8:
                l = parse_maxpool(params);
            break;
            case 9:
                batch_normalize = 1;
                filter = 32;
                size = 1;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "leaky");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);            
            break;
            case 10:
                batch_normalize = 1;
                filter = 256;
                size = 3;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "leaky");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);            
            break;
            case 11:
                batch_normalize = 1;
                filter = 32;
                size = 1;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "leaky");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);            
            break;
            case 12:
                batch_normalize = 1;
                filter = 256;
                size = 3;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "leaky");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);            
            break;
            case 13:
                l = parse_maxpool(params);
            break;
            case 14:
                batch_normalize = 1;
                filter = 64;
                size = 1;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "leaky");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);            
            break;
            case 15:
                batch_normalize = 1;
                filter = 512;
                size = 3;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "leaky");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);            
            break;
            case 16:
                batch_normalize = 1;
                filter = 64;
                size = 1;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "leaky");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);            
            break;
            case 17:
                batch_normalize = 1;
                filter = 512;
                size = 3;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "leaky");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);            
            break;
            case 18:
                batch_normalize = 1;
                filter = 128;
                size = 1;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "leaky");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);            
            break;
            case 19:
                batch_normalize = 0;
                filter = 1000;
                size = 1;
                stride = 1;
                pad = 1;
                strcpy(activation_s, "linear");
                l = parse_convolutional(batch_normalize, filter, size, stride, pad, activation_s, params);            
            break;
            case 20:
                l = parse_avgpool(params);
            break;                        
            case 21:
                l = parse_softmax(params);
            break;           
            case 22:
                l = parse_cost(params);
            break;
            default:
                printf("error!!\n");
            break;

        }

        net.layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if (l.inputs > max_inputs) max_inputs = l.inputs;
        if (l.outputs > max_outputs) max_outputs = l.outputs;

        ++count;
        params.h = l.out_h;
        params.w = l.out_w;
        params.c = l.out_c;
        params.inputs = l.outputs;
        
        if (l.bflops > 0) bflops += l.bflops;

        if (l.w > 1 && l.h > 1) {
            avg_outputs += l.outputs;
            avg_counter++;
        }
    }

    net.outputs = get_network_output_size(net);
    net.output = get_network_output(net);
    avg_outputs = avg_outputs / avg_counter;
    fprintf(stderr, "Total BFLOPS %5.3f \n", bflops);
    fprintf(stderr, "avg_outputs = %d \n", avg_outputs);
    if (workspace_size) {
        net.workspace = (float*)xcalloc(1, workspace_size);
    }

    return net;
}

int load_convolutional_weights(int count, layer l)
{
    int num = l.nweights;
    int i;
    
    for(i = 0; i < l.n; i++){
        l.biases[i] = weight_file[count];
        count++;
    }
    
    if (l.batch_normalize){        
        for(i = 0; i < l.n; i++){
            l.scales[i] = weight_file[count];
            count++;
        }
        for(i = 0; i < l.n; i++){
            l.rolling_mean[i] = weight_file[count];
            count++;
        }
        for(i = 0; i < l.n; i++){
            l.rolling_variance[i] = weight_file[count];
            count++;
        }     
    }

    for(i = 0; i < num; i++){
        l.weights[i] = weight_file[count];
        count++;
    }
    

   return count;
}



void load_weights_upto(network *net, int cutoff)
{
    fprintf(stderr, "Loading weights from...");

    printf("\n seen 32");
    uint32_t iseen = 32204800000;
    *net->seen = iseen;

    *net->cur_iteration = get_current_batch(*net);
    printf(", trained: %.0f K-images (%.0f Kilo-batches_64) \n", (float)(*net->seen / 1000), (float)(*net->seen / 64000));

    int i;
    int count = 0;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            count = load_convolutional_weights(count, l);
        }
    }
    fprintf(stderr, "Done! Loaded %d layers from weights-file \n", i);
}

void load_weights(network *net)
{
    load_weights_upto(net, net->n);
}

