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

#define SECRET_NUM -1234

typedef enum { UNUSED_DEF_VAL } UNUSED_ENUM_TYPE;

#ifdef GPU

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#ifdef CUDNN
#include <cudnn.h>
#endif  // CUDNN
#endif  // GPU

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

struct load_args;
typedef struct load_args load_args;

struct data;
typedef struct data data;

struct metadata;
typedef struct metadata metadata;

struct tree;
typedef struct tree tree;

// option_list.h
typedef struct metadata {
    int classes;
    char **names;
} metadata;


// tree.h
typedef struct tree {
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;


// activations.h
typedef enum {
    LOGISTIC, RELU, RELU6, RELIE, LINEAR, RAMP, TANH, PLSE, REVLEAKY, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU, GELU, SWISH, MISH, HARD_MISH, NORM_CHAN, NORM_CHAN_SOFTMAX, NORM_CHAN_SOFTMAX_MAXVAL
}ACTIVATION;

// parser.h
typedef enum {
    IOU, GIOU, MSE, DIOU, CIOU
} IOU_LOSS;

// parser.h
typedef enum {
    DEFAULT_NMS, GREEDY_NMS, DIOU_NMS, CORNERS_NMS
} NMS_KIND;

// parser.h
typedef enum {
    NO_WEIGHTS, PER_FEATURE, PER_CHANNEL
} WEIGHTS_TYPE_T;

// parser.h
typedef enum {
    NO_NORMALIZATION, RELU_NORMALIZATION, SOFTMAX_NORMALIZATION
} WEIGHTS_NORMALIZATION_T;

// image.h
typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;

// activations.h
typedef enum{
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

// blas.h
typedef struct contrastive_params {
    float sim;
    float exp_sim;
    float P;
    int i, j;
    int time_step_i, time_step_j;
} contrastive_params;


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
    REGION,
    YOLO,
    GAUSSIAN_YOLO,
    ISEG,
    LOGXENT,
    L2NORM,
    EMPTY,
    BLANK,
    IMPLICIT
} LAYER_TYPE;

// layer.h
typedef enum{
    SSE, MASKED, L1, SEG, SMOOTH,WGAN
} COST_TYPE;

// layer.h
typedef struct update_args {
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

// layer.h
struct layer {
    LAYER_TYPE type;
    ACTIVATION activation;
    ACTIVATION lstm_activation;
    COST_TYPE cost_type;
    void(*forward)   (struct layer, struct network_state);
    int train;
    int avgpool;
    int batch_normalize;
    int shortcut;
    int batch;
    int inputs;
    int outputs;
    int nweights;
    int extra;
    int truths;
    int h, w, c;
    int out_h, out_w, out_c;
    int n;
    int groups;
    int group_id;
    int size;
    int side;
    int stride;
    int dilation;
    int maxpool_depth;
    int out_channels;
    float reverse;
    int pad;
    int sqrt;
    int index;
    int scale_wh;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    int rotate;
    float angle;
    float jitter;
    float resize;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    int softmax;
    int classes;
    int detection;
    int coords;
    int rescore;
    int objectness;
    int does_cost;
    int joint;
    int log;
    int tanh;
    int *mask;
    int total;
    float bflops;

    int t;

    float alpha;
    float beta;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    float random;
    float ignore_thresh;
    float truth_thresh;
    float iou_thresh;
    float thresh;
    int absolute;

    int dontsave;
    int numload;

    float temperature;
    float probability;
    int dropblock;
    float scale;

    char  * cweights;
    int   * indexes;
    float **layers_delta;
    WEIGHTS_TYPE_T weights_type;
    WEIGHTS_NORMALIZATION_T weights_normalization;
    int   * map;
    int   * counts;
    float ** sums;
    float * rand;
    float * cost;
    int *labels;
    int *class_ids;
    float * state;
    float * prev_state;

    float *biases;

    float *scales;

    float *weights;

    float scale_x_y;
    int objectness_smooth;
    int new_coords;
    float max_delta;
    float iou_normalizer;
    float obj_normalizer;
    float cls_normalizer;
    IOU_LOSS iou_loss;
    NMS_KIND nms_kind;
    float beta_nms;

    float * delta;
    float * output;
    float * activation_input;
    float * loss;
    float * squared;
    float * norms;

    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    float * m;
    float * v;

    tree *softmax_tree;

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
    float epoch;
    int subdivisions;
    layer *layers;
    float *output;
    learning_rate_policy policy;
    int benchmark_layers;
    int *total_bbox;
    int *rewritten_bbox;

    float learning_rate;
    float learning_rate_max;
    float momentum;
    float decay;
    float gamma;
    float scale;
    int time_steps;
    int step;
    int max_batches;
    int num_boxes;
    float *seq_scales;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;
    int cudnn_half;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    int flip; // horizontal flip 50% probability augmentaiont for classifier training (default = 1)
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;
    int sequential_subdivisions;
    int init_sequential_subdivisions;
    int current_subdivision;

    tree *hierarchy;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;

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

// box.h
typedef struct box {
    float x, y, w, h;
} box;

// box.h
typedef struct boxabs {
    float left, right, top, bot;
} boxabs;

// box.h
typedef struct dxrep {
    float dt, db, dl, dr;
} dxrep;

// box.h
typedef struct ious {
    float iou, giou, diou, ciou;
    dxrep dx_iou;
    dxrep dx_giou;
} ious;


// matrix.h
typedef struct matrix {
    int rows, cols;
    float **vals;
} matrix;

// data.h
typedef struct data {
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

// data.h
typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA
} data_type;

// data.h
typedef struct load_args {
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int c; // color depth
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int scale;
    int center;
    int coords;
    int mini_batch;
    int show_imgs;
    float jitter;
    float resize;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
} load_args;

// parser.c
LIB_API void free_network(network net);

// network.h
LIB_API float *network_predict(network net, float *input);
LIB_API void fuse_conv_batchnorm(network net);

// image.h
LIB_API image resize_image(image im, int w, int h);
LIB_API image make_image(int w, int h, int c);
LIB_API image load_image_color(char *filename, int w, int h);
LIB_API void free_image(image m);
LIB_API image crop_image(image im, int dx, int dy, int w, int h);
LIB_API image resize_min(image im, int min);

// layer.h
LIB_API void free_layer_custom(layer l, int keep_cudnn_desc);
LIB_API void free_layer(layer l);


// utils.h
LIB_API void top_k(float *a, int n, int k, int *index);

// option_list.h
LIB_API metadata get_metadata(char *file);


// http_stream.h
LIB_API double get_time_point();


// gemm.h
LIB_API void init_cpu();

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // DARKNET_API
