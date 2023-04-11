#ifndef DARKNET_API
#define DARKNET_API

#if defined(_MSC_VER) && _MSC_VER < 1900
#define inline __inline
#endif

#if defined(DEBUG) && !defined(_CRTDBG_MAP_ALLOC)
#define _CRTDBG_MAP_ALLOC
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#ifndef LIB_API
#ifdef LIB_EXPORTS
#if defined(_MSC_VER)
#define LIB_API __declspec(dllexport)
#else
#define LIB_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define LIB_API
#else
#define LIB_API
#endif
#endif
#endif


#ifdef __cplusplus
extern "C" {
#endif

struct network;
typedef struct network network;

struct network_state;
typedef struct network_state network_state;

struct layer;
typedef struct layer layer;

struct image;
typedef struct image image;

// activations.h
typedef enum {
    LINEAR, LEAKY, LOGISTIC
}ACTIVATION;

// layer.h
typedef enum {
    CONVOLUTIONAL,
    MAXPOOL,
    LOCAL_AVGPOOL,
    SOFTMAX,
    COST,
    NORMALIZATION,
    AVGPOOL,
    ACTIVE,
    BATCHNORM,
    NETWORK,
    LOGXENT,
    L2NORM,
    EMPTY,
    BLANK,
} LAYER_TYPE;

// layer.h
typedef enum{
    SSE
} COST_TYPE;

// layer.h
struct layer {
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void(*forward)   (struct layer, struct network_state);
    int batch_normalize;
    int batch;
    int inputs;
    int outputs;
    int nweights;
    int h, w, c;
    int out_h, out_w, out_c;
    int n;
    int groups;
    int size;
    int side;
    int stride;
    int pad;
    int classes;
    int total;
    float bflops;

    float scale;

    int   * indexes;
    float * cost;

    float *biases;

    float *scales;

    float * delta;

    float *weights;

    float * output;

    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    size_t workspace_size;

};


// network.h
typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM, SGDR
} learning_rate_policy;

// network.h
typedef struct network {
    int n;
    int batch;
    uint64_t *seen;
    int *cur_iteration;
    int *t;
    int subdivisions;
    layer *layers;
    float *output;
    learning_rate_policy policy;
    int *total_bbox;
    int *rewritten_bbox;

    int time_steps;
    int step;
    int max_batches;

    int inputs;
    int outputs;
    int h, w, c;

    float *input;
    float *workspace;
    int train;

    size_t workspace_size_limit;
} network;

// network.h
typedef struct network_state {
    float *truth;
    float *input;
    float *delta;
    float *workspace;
    int train;
    int index;
    network net;
} network_state;


// image.h
typedef struct image {
    int w;
    int h;
    int c;
    float *data;
} image;

// matrix.h
typedef struct matrix {
    int rows, cols;
    float **vals;
} matrix;

// parser.c
LIB_API void free_network(network net);

// network.h
LIB_API float *network_predict(network net, float *input);
LIB_API void fuse_conv_batchnorm(network net);

// image.h
LIB_API image resize_image(image im, int w, int h);
LIB_API image make_image(int w, int h, int c);
LIB_API image load_image_color(int w, int h);
LIB_API void free_image(image m);
LIB_API image crop_image(image im, int dx, int dy, int w, int h);
LIB_API image resize_min(image im, int min);

// layer.h
LIB_API void free_layer_custom(layer l, int keep_cudnn_desc);
LIB_API void free_layer(layer l);


// utils.h
LIB_API void top_k(float *a, int n, int k, int *index);

// http_stream.h
LIB_API double get_time_point();

// gemm.h
LIB_API void init_cpu();

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // DARKNET_API
