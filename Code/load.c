#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "CNN.h"
#include "load.h"


struct Parameter params[NUM_PARAMS];

/**
 * @brief This function puts the weights associated with BIN_PATH/file_name inside
 * of the passed buffer, which has a length of size. Must be appropriately sized.
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
 * Multiplies the elements of the shape to get the len of the associated tensor
 */
static int get_size(struct Shape shape) {
    int size = 1;
    for (int i = 0; i < shape.len; i++) {
        size *= shape.dim[i];
    }
    return size;
}

/**
 * Loops over the params array to initialize each of the weights
 * Basically just a wrapper around put_weights
 */
static void init_weights() {
    for (int i = 0; i < NUM_PARAMS; i++) {
        struct Parameter parameter = params[i];
        char *file_name = parameter.filename;
        put_weights(file_name, parameter.tensor.data, parameter.tensor.len);
    }
}

/**
 * Initializes a Parameter with the proper specification based on the arguments
 */
static void init_param(int idx, int *dims_array, int size, char *filename) {
    int *dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, dims_array, sizeof(int) * size);
    struct Shape shape = {dims, size};
    struct Tensor tensor = construct_tensor(shape);
    params[idx] = (struct Parameter) {.tensor = tensor, .filename = filename};
}


/**
 * Initializes each element of the params array with the appropriate specification
 * Hardcoded based on our a priori knowledge of the parameters from the python program
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
 * Prints the weights from the params array. Allows us to verify correctness against python
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
 * Loads the given batch into the provided data Tensor
 * The parameter batch_num should be 0 indexed
 */
void load_batch(int batch_num, struct Tensor *data) {
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
    *data = construct_tensor(shape);
    char *file_name = "test_data.bin";
    
    // Concatenate the path of our binary
    char *new_buf = (char *) malloc(strlen(BIN_PATH) + strlen(file_name) + 1);
    memcpy(new_buf, BIN_PATH, strlen(BIN_PATH));
    memcpy(new_buf + strlen(BIN_PATH), file_name, strlen(file_name) + 1);

    FILE *f = fopen(new_buf, "rb");

    // Sanity checking:
    fseek(f, 0, SEEK_END);
    int f_size = ftell(f);
    int exp_size = (128 * 200 * 5 * 5) * (NUM_BATCHES - 1) * sizeof(float) + (5 * 200 * 5 * 5 * sizeof(float));
    assert(f_size == exp_size);
    rewind(f);
    
    // Read the data
    fseek(f, (*data).len * sizeof(float) * batch_num, SEEK_SET);
    fread((*data).data, sizeof(float), get_size(shape), f);
    fclose(f);
    
    free(new_buf);
}

/**
 * Does the full initialization
 */
void full_weight_init() {
    init_params();
    init_weights();
}

/**
 * Frees the dynamically allocated resources of a tensor
 */
void destroy_tensor(struct Tensor *data) {
    free(data->shape.dim);
    free(data->data);
    free(data->prefixes);
}

/**
 * Does the computation of prefixes associated with a shape
 * Prefixes are useful for calculating indexes, thus it is useful to store them
 * As a part of our tensor
 */
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
 * Constructs a tensor based on the passed shape parameter
 * The tensor uses the parameter shape itself in its shape field, so one should not reuse
 * shapes to create multiple tensors, otherwise their dimension arrays will be shared
 * This is all possible because everything about a tensor is uniquely determined by its shape
 * (Except its data values which will need to be filled in by the caller)
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

/**
 * Stores the data from the tensor parameter into a file with path BIN_PATH/filename
 */
void put_batch(struct Tensor *data, char *filename) {
    char *new_buf = (char *) malloc(strlen(BIN_PATH) + strlen(filename) + 1);
    memcpy(new_buf, BIN_PATH, strlen(BIN_PATH));
    memcpy(new_buf + strlen(BIN_PATH), filename, strlen(filename) + 1);

    // I believe fopen would already destroy the file if it existed, but this is safer
    remove(new_buf);

    FILE *f = fopen(new_buf, "wb");
    assert(f);

    int w = fwrite(data->data, sizeof(float), data->len, f);
    assert(w == data->len);

    fclose(f);
    free(new_buf);
}
