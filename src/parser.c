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

typedef struct{
    char *type;
    list *options;
}section;

list *read_cfg(char *filename);

LAYER_TYPE string_to_layer_type(char * type)
{
    if (strcmp(type, "[cost]")==0) return COST;
    if (strcmp(type, "[conv]")==0
            || strcmp(type, "[convolutional]")==0) return CONVOLUTIONAL;
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return NETWORK;
    if (strcmp(type, "[max]")==0
            || strcmp(type, "[maxpool]")==0) return MAXPOOL;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX;
    return BLANK;
}

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

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


convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int groups = option_find_int_quiet(options, "groups", 1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);

    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    convolutional_layer layer = make_convolutional_layer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize);

    return layer;
}

softmax_layer parse_softmax(list *options, size_params params)
{
	int groups = option_find_int_quiet(options, "groups", 1);
	softmax_layer layer = make_softmax_layer(params.batch, params.inputs, groups);
    //remove read_tree
	layer.w = params.w;
	layer.h = params.h;
	layer.c = params.c;
	return layer;
}

cost_layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    cost_layer layer = make_cost_layer(params.batch, params.inputs, type);
    return layer;
}


maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);
    const int avgpool = 0;

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;

    maxpool_layer layer = make_maxpool_layer(batch, h, w, c, size, stride, padding, avgpool);
    return layer;
}

avgpool_layer parse_avgpool(list *options, size_params params)
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

void parse_net_options(list *options, network *net)
{
    net->max_batches = option_find_int(options, "max_batches", 0);
    net->batch = option_find_int(options, "batch",1);
    int subdivs = option_find_int(options, "subdivisions",1);
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->batch /= subdivs;          // mini_batch
    net->batch *= net->time_steps;  // mini_batch * time_steps
    net->subdivisions = subdivs;    // number of mini_batches

    *net->seen = 0;
    *net->cur_iteration = 0;
    net->workspace_size_limit = (size_t)1024*1024 * option_find_float_quiet(options, "workspace_size_limit_MB", 1024);  // 1024 MB by default

    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);

}

int is_network(section *s)
{
    return (strcmp(s->type, "[net]")==0
            || strcmp(s->type, "[network]")==0);
}


network parse_network_cfg(char *filename)
{
    return parse_network_cfg_custom(filename, 0, 0);
}

network parse_network_cfg_custom(char *filename, int batch, int time_steps)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) error("Config file has no sections", DARKNET_LOC);
    network net = make_network(sections->size - 1);
    size_params params;

    params.train = 0;    // allocates memory for Inference only

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) error("First section must be [net] or [network]", DARKNET_LOC);
    parse_net_options(options, &net);

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
    printf("mini_batch = %d, batch = %d, time_steps = %d, train = %d \n", net.batch, net.batch * net.subdivisions, net.time_steps, params.train);

    int last_stop_backward = -1;
    int avg_outputs = 0;
    int avg_counter = 0;
    float bflops = 0;
    size_t workspace_size = 0;
    size_t max_inputs = 0;
    size_t max_outputs = 0;

    n = n->next;
    int count = 0;
    free_section(s);

    int old_params_train = params.train;

    fprintf(stderr, "   layer   filters  size/strd(dil)      input                output\n");
    while(n){

        params.train = old_params_train;
        if (count < last_stop_backward) params.train = 0;

        params.index = count;
        fprintf(stderr, "%4d ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = { (LAYER_TYPE)0 };
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional(options, params);
        }else if(lt == COST){
            l = parse_cost(options, params);
        }else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
        }else if(lt == MAXPOOL){
            l = parse_maxpool(options, params);
        }else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }

        // remove calculate receptive field

        option_unused(options);


        net.layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if (l.inputs > max_inputs) max_inputs = l.inputs;
        if (l.outputs > max_outputs) max_outputs = l.outputs;
        free_section(s);
        n = n->next;
        ++count;
        if(n){
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
        if (l.bflops > 0) bflops += l.bflops;

        if (l.w > 1 && l.h > 1) {
            avg_outputs += l.outputs;
            avg_counter++;
        }
    }

    free_list(sections);

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



list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *sections = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = (section*)xmalloc(sizeof(section));
                list_insert(sections, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return sections;
}

void load_convolutional_weights(layer l, FILE *fp)
{

    int num = l.nweights;
    int read_bytes;
    read_bytes = fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        read_bytes = fread(l.scales, sizeof(float), l.n, fp);
        read_bytes = fread(l.rolling_mean, sizeof(float), l.n, fp);
        read_bytes = fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    read_bytes = fread(l.weights, sizeof(float), num, fp);

}



void load_weights_upto(network *net, char *filename, int cutoff)
{

    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major * 10 + minor) >= 2) {
        printf("\n seen 64");
        uint64_t iseen = 0;
        fread(&iseen, sizeof(uint64_t), 1, fp);
        *net->seen = iseen;
    }
    else {
        printf("\n seen 32");
        uint32_t iseen = 0;
        fread(&iseen, sizeof(uint32_t), 1, fp);
        *net->seen = iseen;
    }
    *net->cur_iteration = get_current_batch(*net);
    printf(", trained: %.0f K-images (%.0f Kilo-batches_64) \n", (float)(*net->seen / 1000), (float)(*net->seen / 64000));

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (0) continue;
        if(l.type == CONVOLUTIONAL){
            load_convolutional_weights(l, fp);
        }
        if (feof(fp)) break;
    }
    fprintf(stderr, "Done! Loaded %d layers from weights-file \n", i);
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, net->n);
}

