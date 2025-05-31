#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "CNN.h"
#include "load.h"

#define BIN_PATH "../Weight_Binaries/"

struct Parameter params[NUM_PARAMS];
struct Tensor g_data;

/**
 * @brief This function puts the weights associated with file_name.bin inside
 * of the appropriately sized buffer
 * Note that the size field is only passed as a sanity check for the size of our binaries
 */
static void put_weights(char *file_name, float *buf, int size) {
    char *new_buf = (char *) malloc(strlen(BIN_PATH) + strlen(file_name) + 1);
    memcpy(new_buf, BIN_PATH, strlen(BIN_PATH));
    memcpy(new_buf + strlen(BIN_PATH), file_name, strlen(file_name) + 1);

    FILE *f = fopen(new_buf, "rb");
    fseek(f, 0, SEEK_END);
    int alt_size = ftell(f) / sizeof(float);
    assert(alt_size == size);
    rewind(f);

    fread(buf, sizeof(float), size, f);

    fclose(f);
    free(new_buf);
}

/**
 * @brief helper function which multiplies the elements of the shape to get the size of the weights arr
 */
static int get_size(struct Shape shape) {
    int size = 1;
    for (int i = 0; i < shape.len; i++) {
        size *= shape.dim[i];
    }
    return size;
}

/**
 * @brief Function which loops over the params array to initialize each of the weights
 */
static void init_weights() {
    for (int i = 0; i < NUM_PARAMS; i++) {
        struct Parameter parameter = params[i];
        char *file_name = parameter.filename;
        put_weights(file_name, parameter.tensor.data, parameter.tensor.len);
    }
}

/**
 * @brief Helper function which initializes a Parameter with the proper specification
 */
static void init_param(int idx, int *dims_array, int size, char *filename) {
    int *dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, dims_array, sizeof(int) * size);
    struct Shape shape = {dims, size};
    struct Tensor tensor = construct_tensor(shape);
    params[idx] = (struct Parameter) {.tensor = tensor, .filename = filename};
}


// Sets up the Parameter fields of the params array
/**
 * @brief Initializes each element of the params array with the appropriate specification
 * Uses the above helper function
 */
static void init_params() {
    init_param(CONV10W_IDX, (int[]){16, 1, 3, 3, 3}, 5, "conv_layer1.0.weight.bin");
    init_param(CONV10B_IDX, (int[]){16}, 1, "conv_layer1.0.bias.bin");

    init_param(CONV12W_IDX, (int[]){16}, 1, "conv_layer1.2.weight.bin");
    init_param(CONV12B_IDX, (int[]){16}, 1, "conv_layer1.2.bias.bin");
    init_param(CONV12MEAN_IDX, (int[]){16}, 1, "conv_layer1.2.running_mean.bin");
    init_param(CONV12VAR_IDX, (int[]){16}, 1, "conv_layer1.2.running_var.bin");

    init_param(CONV20W_IDX, (int[]){16, 16, 3, 3, 3}, 5, "conv_layer2.0.weight.bin");
    init_param(CONV20B_IDX, (int[]){16}, 1, "conv_layer2.0.bias.bin");

    init_param(CONV22W_IDX, (int[]){16}, 1, "conv_layer2.2.weight.bin");
    init_param(CONV22B_IDX, (int[]){16}, 1, "conv_layer2.2.bias.bin");
    init_param(CONV22MEAN_IDX, (int[]){16}, 1, "conv_layer2.2.running_mean.bin");
    init_param(CONV22VAR_IDX, (int[]){16}, 1, "conv_layer2.2.running_var.bin");

    init_param(SCONV10W_IDX, (int[]){3200, 1, 5, 5}, 4, "sepconv1.0.weight.bin");
    init_param(SCONV10B_IDX, (int[]){3200}, 1, "sepconv1.0.bias.bin");

    init_param(SCONV12W_IDX, (int[]){320, 3200, 1, 1}, 4, "sepconv1.2.weight.bin");
    init_param(SCONV12B_IDX, (int[]){320}, 1, "sepconv1.2.bias.bin");

    init_param(SCONV14W_IDX, (int[]){320}, 1, "sepconv1.4.weight.bin");
    init_param(SCONV14B_IDX, (int[]){320}, 1, "sepconv1.4.bias.bin");
    init_param(SCONV14MEAN_IDX, (int[]){320}, 1, "sepconv1.4.running_mean.bin");
    init_param(SCONV14VAR_IDX, (int[]){320}, 1, "sepconv1.4.running_var.bin");

    init_param(SCONV20W_IDX, (int[]){320, 1, 3, 3}, 4, "sepconv2.0.weight.bin");
    init_param(SCONV20B_IDX, (int[]){320}, 1, "sepconv2.0.bias.bin");

    init_param(SCONV22W_IDX, (int[]){256, 320, 1, 1}, 4, "sepconv2.2.weight.bin");
    init_param(SCONV22B_IDX, (int[]){256}, 1, "sepconv2.2.bias.bin");

    init_param(SCONV24W_IDX, (int[]){256}, 1, "sepconv2.4.weight.bin");
    init_param(SCONV24B_IDX, (int[]){256}, 1, "sepconv2.4.bias.bin");
    init_param(SCONV24MEAN_IDX, (int[]){256}, 1, "sepconv2.4.running_mean.bin");
    init_param(SCONV24VAR_IDX, (int[]){256}, 1, "sepconv2.4.running_var.bin");

    init_param(SCONV30W_IDX, (int[]){256, 1, 3, 3}, 4, "sepconv3.0.weight.bin");
    init_param(SCONV30B_IDX, (int[]){256}, 1, "sepconv3.0.bias.bin");

    init_param(SCONV32W_IDX, (int[]){256, 256, 1, 1}, 4, "sepconv3.2.weight.bin");
    init_param(SCONV32B_IDX, (int[]){256}, 1, "sepconv3.2.bias.bin");

    init_param(SCONV34W_IDX, (int[]){256}, 1, "sepconv3.4.weight.bin");
    init_param(SCONV34B_IDX, (int[]){256}, 1, "sepconv3.4.bias.bin");
    init_param(SCONV34MEAN_IDX, (int[]){256}, 1, "sepconv3.4.running_mean.bin");
    init_param(SCONV34VAR_IDX, (int[]){256}, 1, "sepconv3.4.running_var.bin");

    init_param(FC1W_IDX, (int[]){16, 256}, 2, "fc1.weight.bin");
    init_param(FC1B_IDX, (int[]){16}, 1, "fc1.bias.bin");
}

/**
 * @brief prints the weights in the params array so we can check against the weights from python
 */
void print_weights() {
    for (int i = 0; i < NUM_PARAMS; i++) {
        struct Parameter parameter = params[i];
        printf("Parameter name: %s\n", parameter.filename);
        for (int j = 0; j < parameter.tensor.len; j++) {
            printf("%f ", parameter.tensor.data[j]);
        }
        printf("\n\n\n");
    }
}

/**
 * @brief Loads the given batch into the global data variable
 * Note that the parameter batch_num should be 0 indexed
 */
void load_batch(int batch_num) {
    int *dims = (int *) malloc(sizeof(int) * 5);
    struct Shape shape = {dims, 5};
    dims[0] = 128;
    dims[1] = 1;
    dims[2] = 200;
    dims[3] = 5;
    dims[4] = 5;
    /* int size = get_size(shape); // num of entries in a normal batch */
    if (batch_num == NUM_BATCHES - 1) {
        dims[0] = 5;
    }
    g_data = construct_tensor(shape);
    char *file_name = "test_data.bin";
    
    // Concatenate the path of our binary
    char *new_buf = (char *) malloc(strlen(BIN_PATH) + strlen(file_name) + 1);
    memcpy(new_buf, BIN_PATH, strlen(BIN_PATH));
    memcpy(new_buf + strlen(BIN_PATH), file_name, strlen(file_name) + 1);

    FILE *f = fopen(new_buf, "rb");

    // Sanity checking:
    fseek(f, 0, SEEK_END);
    int f_size = ftell(f);
    int exp_size = g_data.len * (NUM_BATCHES - 1) * sizeof(float) + (5 * 200 * 5 * 5 * sizeof(float));
    assert(f_size == exp_size);
    rewind(f);
    
    fseek(f, g_data.len * sizeof(float) * batch_num, SEEK_SET);
    fread(g_data.data, sizeof(float), get_size(shape), f);
    fclose(f);
    
    free(new_buf);
}

/**
 * @brief Does the full initialization. We don't want to expose the other functions out of this file
 */
void full_weight_init() {
    init_params();
    init_weights();
}

void destroy_tensor(struct Tensor data) {
    free(data.shape.dim);
    free(data.data);
    free(data.prefixes);
}

// Note that I don't think this works right now
/**
 * @brief Moves src into dest, freeing any memory associated with dest before
 */
void move_tensor(struct Tensor dest, struct Tensor src) {
    destroy_tensor(dest);
    dest = src;
}

int *compute_prefixes(struct Shape shape) {
    int dim_len = shape.len;
    int *dimensions = shape.dim;
    int *prefixes = (int *) malloc(sizeof(int) * dim_len);

    for (int i = dim_len - 1; i >= 0; i--) {
        if (i == dim_len - 1) {
            prefixes[i] = 1;
            continue;
        }
        prefixes[i] = prefixes[i + 1] * dimensions[i+1];
    }

    return prefixes;
}

/**
 * @brief Everything about a tensor is uniquely determined by its shape
 * (except actual data values). We construct a tensor with a passed shape
 * the tensor will use shape as its actual shape field, so one should not reuse
 * shapes to create multiple tensors, otherwise their dimension arrays will be shared
 */
struct Tensor construct_tensor(struct Shape shape) {
    int *prefixes = compute_prefixes(shape);
    int length = get_size(shape);
    float *data = (float *) malloc(sizeof(float) * length);
    if (data == NULL) {
        printf("Failed to successfully malloc\n");
        exit(EXIT_FAILURE);
    }
    struct Tensor tens = {
        .shape = shape,
        .data = data,
        .prefixes = prefixes,
        .len = length,
    };
    return tens;
}
