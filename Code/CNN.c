#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "CNN.h"

struct ParamInfo params[NUM_PARAMS];

// Puts the weights associated with file_name.bin inside of the (appropriately sized) buffer
// The size field is passed to make sure our date aligns with reality
void put_weights(char *file_name, float *buf, int size) {
    FILE *f = fopen(file_name, "rb");
    fseek(f, 0, SEEK_END);
    int alt_size = ftell(f) / sizeof(float);
    /* printf("altsize = %d, size = %d\n", alt_size, size); */
    assert(alt_size == size);
    rewind(f);

    fread(buf, sizeof(float), size, f);
    fclose(f);
}

int get_wgt_size(int dim_size, int *dimensions) {
    int size = 1;
    for (int i = 0; i < dim_size; i++) {
        size *= dimensions[i];
    }
    return size;
}

// Initializes the weights of each element of the params array
void init_weights() {
    for (int i = 0; i < NUM_PARAMS; i++) {
        struct ParamInfo parameter = params[i];
        char *file_name = parameter.filename;
        int wgt_size = get_wgt_size(parameter.dim_len, parameter.dimensions);
        /* for (int j = 0; j < parameter.dim_len; j++) { */
        /*     wgt_size *= parameter.dimensions[j]; */
        /* } */
        /* printf("file name: %s\n", parameter.filename); */
        put_weights(file_name, parameter.weights, wgt_size);
    }
}


// Sets up the ParamInfo fields of the params array
void init_params() {
    int size;
    int *dims;
    float *weights;

    // CONV10W
    size = 5;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {16, 1, 3, 3, 3}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 16 * 3 * 3 * 3);
    params[CONV10W_IDX] = (struct ParamInfo){dims, size, weights, "conv_layer1.0.weight.bin"};

    // CONV10B
    size = 1;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {16}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 16);
    params[CONV10B_IDX] = (struct ParamInfo){dims, size, weights, "conv_layer1.0.bias.bin"};

    // CONV12W
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {16}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 16);
    params[CONV12W_IDX] = (struct ParamInfo){dims, size, weights, "conv_layer1.2.weight.bin"};

    // CONV12B
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {16}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 16);
    params[CONV12B_IDX] = (struct ParamInfo){dims, size, weights, "conv_layer1.2.bias.bin"};

    // CONV12MEAN
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {16}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 16);
    params[CONV12MEAN_IDX] = (struct ParamInfo){dims, size, weights, "conv_layer1.2.running_mean.bin"};

    // CONV12VAR
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {16}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 16);
    params[CONV12VAR_IDX] = (struct ParamInfo){dims, size, weights, "conv_layer1.2.running_var.bin"};

    // CONV20W
    size = 5;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {16, 16, 3, 3, 3}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 16 * 16 * 3 * 3 * 3);
    params[CONV20W_IDX] = (struct ParamInfo){dims, size, weights, "conv_layer2.0.weight.bin"};

    // CONV20B
    size = 1;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {16}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 16);
    params[CONV20B_IDX] = (struct ParamInfo){dims, size, weights, "conv_layer2.0.bias.bin"};

    // CONV22W
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {16}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 16);
    params[CONV22W_IDX] = (struct ParamInfo){dims, size, weights, "conv_layer2.2.weight.bin"};

    // CONV22B
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {16}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 16);
    params[CONV22B_IDX] = (struct ParamInfo){dims, size, weights, "conv_layer2.2.bias.bin"};

    // CONV22MEAN
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {16}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 16);
    params[CONV22MEAN_IDX] = (struct ParamInfo){dims, size, weights, "conv_layer2.2.running_mean.bin"};

    // CONV22VAR
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {16}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 16);
    params[CONV22VAR_IDX] = (struct ParamInfo){dims, size, weights, "conv_layer2.2.running_var.bin"};

    // SCONV10W
    size = 4;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {3200, 1, 5, 5}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 3200 * 5 * 5);
    params[SCONV10W_IDX] = (struct ParamInfo){dims, size, weights, "sepconv1.0.weight.bin"};

    // SCONV10B
    size = 1;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {3200}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 3200);
    params[SCONV10B_IDX] = (struct ParamInfo){dims, size, weights, "sepconv1.0.bias.bin"};

    // SCONV12W
    size = 4;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {320, 3200, 1, 1}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 320 * 3200);
    params[SCONV12W_IDX] = (struct ParamInfo){dims, size, weights, "sepconv1.2.weight.bin"};

    // SCONV12B
    size = 1;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {320}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 320);
    params[SCONV12B_IDX] = (struct ParamInfo){dims, size, weights, "sepconv1.2.bias.bin"};

    // SCONV14W
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {320}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 320);
    params[SCONV14W_IDX] = (struct ParamInfo){dims, size, weights, "sepconv1.4.weight.bin"};

    // SCONV14B
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {320}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 320);
    params[SCONV14B_IDX] = (struct ParamInfo){dims, size, weights, "sepconv1.4.bias.bin"};

    // SCONV14MEAN
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {320}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 320);
    params[SCONV14MEAN_IDX] = (struct ParamInfo){dims, size, weights, "sepconv1.4.running_mean.bin"};

    // SCONV14VAR
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {320}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 320);
    params[SCONV14VAR_IDX] = (struct ParamInfo){dims, size, weights, "sepconv1.4.running_var.bin"};

    // SCONV20W
    size = 4;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {320, 1, 3, 3}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 320 * 3 * 3);
    params[SCONV20W_IDX] = (struct ParamInfo){dims, size, weights, "sepconv2.0.weight.bin"};

    // SCONV20B
    size = 1;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {320}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 320);
    params[SCONV20B_IDX] = (struct ParamInfo){dims, size, weights, "sepconv2.0.bias.bin"};

    // SCONV22W
    size = 4;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {256, 320, 1, 1}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 256 * 320);
    params[SCONV22W_IDX] = (struct ParamInfo){dims, size, weights, "sepconv2.2.weight.bin"};

    // SCONV22B
    size = 1;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {256}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 256);
    params[SCONV22B_IDX] = (struct ParamInfo){dims, size, weights, "sepconv2.2.bias.bin"};

    // SCONV24W
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {256}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 256);
    params[SCONV24W_IDX] = (struct ParamInfo){dims, size, weights, "sepconv2.4.weight.bin"};

    // SCONV24B
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {256}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 256);
    params[SCONV24B_IDX] = (struct ParamInfo){dims, size, weights, "sepconv2.4.bias.bin"};

    // SCONV24MEAN
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {256}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 256);
    params[SCONV24MEAN_IDX] = (struct ParamInfo){dims, size, weights, "sepconv2.4.running_mean.bin"};

    // SCONV24VAR
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {256}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 256);
    params[SCONV24VAR_IDX] = (struct ParamInfo){dims, size, weights, "sepconv2.4.running_var.bin"};

    // SCONV30W
    size = 4;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {256, 1, 3, 3}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 256 * 3 * 3);
    params[SCONV30W_IDX] = (struct ParamInfo){dims, size, weights, "sepconv3.0.weight.bin"};

    // SCONV30B
    size = 1;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {256}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 256);
    params[SCONV30B_IDX] = (struct ParamInfo){dims, size, weights, "sepconv3.0.bias.bin"};

    // SCONV32W
    size = 4;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {256, 256, 1, 1}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 256 * 256);
    params[SCONV32W_IDX] = (struct ParamInfo){dims, size, weights, "sepconv3.2.weight.bin"};

    // SCONV32B
    size = 1;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {256}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 256);
    params[SCONV32B_IDX] = (struct ParamInfo){dims, size, weights, "sepconv3.2.bias.bin"};

    // SCONV34W
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {256}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 256);
    params[SCONV34W_IDX] = (struct ParamInfo){dims, size, weights, "sepconv3.4.weight.bin"};

    // SCONV34B
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {256}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 256);
    params[SCONV34B_IDX] = (struct ParamInfo){dims, size, weights, "sepconv3.4.bias.bin"};

    // SCONV34MEAN
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {256}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 256);
    params[SCONV34MEAN_IDX] = (struct ParamInfo){dims, size, weights, "sepconv3.4.running_mean.bin"};

    // SCONV34VAR
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {256}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 256);
    params[SCONV34VAR_IDX] = (struct ParamInfo){dims, size, weights, "sepconv3.4.running_var.bin"};

    // FC1W
    size = 2;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {16, 256}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 16 * 256);
    params[FC1W_IDX] = (struct ParamInfo){dims, size, weights, "fc1.weight.bin"};

    // FC1B
    size = 1;
    dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, (int[]) {16}, sizeof(int) * size);
    weights = (float *) malloc(sizeof(float) * 16);
    params[FC1B_IDX] = (struct ParamInfo){dims, size, weights, "fc1.bias.bin"};
}

void print_weights() {
    for (int i = 0; i < NUM_PARAMS; i++) {
        struct ParamInfo parameter = params[i];
        printf("Parameter name: %s\n", parameter.filename);
        int wgt_size = get_wgt_size(parameter.dim_len, parameter.dimensions);
        for (int j = 0; j < wgt_size; j++) {
            printf("%f ", parameter.weights[j]);
        }
        printf("\n\n\n");
    }
}

int main(int argc, char **argv) {
    init_params();
    init_weights();
    print_weights();
    return EXIT_SUCCESS;
}
