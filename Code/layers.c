#include <stdio.h>
#include <stdlib.h>
#include "Math.h"
#include "load.h"
#include "layers.h"
#include "counter.h"

// This is a wrapper around each math function we apply
// It is useful in case we want to do something at each step, such as printing results or shape
// Or doing some timing/benchmarking
#define APPLY_LAYER(new_result) do {                                            \
        (new_result);                                                           \
        printf("For this layer, the data_add, data_mult, idx_add, idx_mult, and relu local counters are:\n"); \
        printf("data_add: %" PRIu64 ", data_mult: %" PRIu64 ", idx_add: %" PRIu64 ", idx_mult: %" PRIu64 ", relu: %" PRIu64"\n", \
               data_add_ctr, data_mult_ctr, idx_add_ctr, idx_mult_ctr, relu_ctr); \
        printf("For this layer, the global data_add, data_mult, idx_add, idx_mult, and relu counters are:\n"); \
        printf("data_add: %" PRIu64 ", data_mult: %" PRIu64 ", idx_add: %" PRIu64 ", idx_mult: %" PRIu64 ", relu: %" PRIu64"\n", \
               data_add_accum_ctr, data_mult_accum_ctr, idx_add_accum_ctr, idx_mult_accum_ctr, relu_accum_ctr); \
} while(0)

static int img_shape[4] = {1, 200, 5, 5};




// Each of these layers group together math operations in the same way as in
// the layers inside of CNN.py

void conv_layer1(struct Tensor *data) {
    printf("\n\nInside of conv_layer1\n");
    APPLY_LAYER(Conv3d(params[CONV10W_IDX], params[CONV10B_IDX], data, 16, 3, 1, 1, 1));
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(BatchNorm3d(params[CONV12W_IDX], params[CONV12B_IDX], params[CONV12MEAN_IDX], params[CONV12VAR_IDX], data));
}

void conv_layer2(struct Tensor *data) {
    printf("\n\nInside of conv_layer2\n");
    APPLY_LAYER(Conv3d(params[CONV20W_IDX], params[CONV20B_IDX], data, 16, 3, 1, 1, 1));
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(BatchNorm3d(params[CONV22W_IDX], params[CONV22B_IDX], params[CONV22MEAN_IDX], params[CONV22VAR_IDX], data));
}

void sepconv1(struct Tensor *data) {
    printf("\n\nInside of sepconv1\n");
    APPLY_LAYER(DepthwiseConv2d(params[SCONV10W_IDX], params[SCONV10B_IDX], data, 5, 2, 1, 1));
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(Conv2d(params[SCONV12W_IDX], params[SCONV12B_IDX], data, 320, 1, 0, 1, 1));
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(BatchNorm2d(params[SCONV14W_IDX], params[SCONV14B_IDX], params[SCONV14MEAN_IDX], params[SCONV14VAR_IDX], data));
}

void sepconv2(struct Tensor *data) {
    printf("\n\nInside of sepconv2\n");
    APPLY_LAYER(DepthwiseConv2d(params[SCONV20W_IDX], params[SCONV20B_IDX], data, 3, 1, 1, 1));
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(Conv2d(params[SCONV22W_IDX], params[SCONV22B_IDX], data, 256, 1, 0, 1, 1));
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(BatchNorm2d(params[SCONV24W_IDX], params[SCONV24B_IDX], params[SCONV24MEAN_IDX], params[SCONV24VAR_IDX], data));
}

void sepconv3(struct Tensor *data) {
    printf("\n\nInside of sepconv3\n");
    APPLY_LAYER(DepthwiseConv2d(params[SCONV30W_IDX], params[SCONV30B_IDX], data, 3, 1, 1, 1));
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(Conv2d(params[SCONV32W_IDX], params[SCONV32B_IDX], data, 256, 1, 0, 1, 1));
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(BatchNorm2d(params[SCONV34W_IDX], params[SCONV34B_IDX], params[SCONV34MEAN_IDX], params[SCONV34VAR_IDX], data));
}

void average(struct Tensor *data) {
    printf("\n\nInside of average\n");
    APPLY_LAYER(AvgPool2d(data, 5));
}

void fc1(struct Tensor *data) {
    printf("\n\nInside of fc1\n");
    APPLY_LAYER(Linear(256, 16, params[FC1W_IDX], params[FC1B_IDX], data));
}


// These reshape operations correspond to the reshaping done in CNN.py

void reshape1(struct Tensor *data) {
    int x_shape_0 = data->shape.dim[0];
    free(data->shape.dim);

    data->shape.dim = (int *) malloc(sizeof(int) * 4);
    data->shape.len = 4;
    data->shape.dim[0] = x_shape_0;
    data->shape.dim[1] = img_shape[1] * 16;
    data->shape.dim[2] = img_shape[2];
    data->shape.dim[3] = img_shape[3];
    free(data->prefixes);
    data->prefixes = compute_prefixes(data->shape);
}

void reshape2(struct Tensor *data) {
    int x_shape_0 = data->shape.dim[0];
    int x_shape_1 = data->shape.dim[1];
    free(data->shape.dim);

    data->shape.dim = (int *) malloc(sizeof(int) * 2);
    data->shape.len = 2;
    data->shape.dim[0] = x_shape_0;
    data->shape.dim[1] = x_shape_1;
    free(data->prefixes);
    data->prefixes = compute_prefixes(data->shape);
}
