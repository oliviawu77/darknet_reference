#include "network.h"
#include "utils.h"
#include "parser.h"
#include "blas.h"
#include "assert.h"
#include "classifier.h"
#ifdef WIN32
#include <time.h>
#else
#include <sys/time.h>
#endif

void predict_classifier(int top)
{
    network net = parse_network_cfg_custom(1, 0);
    load_weights(&net);
    set_batch_network(&net, 1);
    srand(2222222);

    fuse_conv_batchnorm(net);

    char *name_list;
    name_list = (char*)malloc(sizeof(char)* 512);
    strcpy(name_list, "data/imagenet.shortnames.list");

    int classes = 1000;
    printf(" classes = %d, output in cfg = %d \n", classes, net.layers[net.n - 1].c);
    layer l = net.layers[net.n - 1];
    if (classes != l.outputs && (l.type == SOFTMAX || l.type == COST)) {
        printf("\n Error: num of filters = %d in the last conv-layer in cfg-file doesn't match to classes = %d in data-file \n",
            l.outputs, classes);
        getchar();
    }
    top = 5;
    if (top > classes) top = classes;

    int i = 0;
    int* indexes = (int*)xcalloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    double begin = get_time_point();
        
    image im = load_image_color(0, 0);
    image resized = resize_min(im, net.w);
    image cropped = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
    printf("%d %d\n", cropped.w, cropped.h);

    float *X = cropped.data;

    double time = get_time_point();
    float *predictions = network_predict(net, X);
    printf("%s: Predicted in %lf milli-seconds.\n", "dog", ((double)get_time_point() - time) / 1000);

    top_k(predictions, net.outputs, top, indexes);

    for(i = 0; i < top; ++i){
        int index = indexes[i];
        printf("%s: %f\n",names[index], predictions[index]);
    }

    free_image(cropped);
    if (resized.data != im.data) {
        free_image(resized);
    }
    free_image(im);

    double end = get_time_point();
    printf("Executing: %lf milli-seconds.\n", (end - begin) / 1000);
    
    free(indexes);
    free_network(net);
}


