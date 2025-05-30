/* #include <cstdint> */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "CNN.h"
#include "load_params.h"

#define BIN_PATH "../Weight_Binaries/"

struct ParamInfo params[NUM_PARAMS];
struct ParamInfo data;

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
int get_wgt_size(struct Shape shape) {
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
        struct ParamInfo parameter = params[i];
        char *file_name = parameter.filename;
        int wgt_size = get_wgt_size(parameter.shape);
        put_weights(file_name, parameter.weights, wgt_size);
    }
}

/**
 * @brief Helper function which initializes a ParamInfo with the proper specification
 */
static void init_param(int idx, int *dims_array, int size, size_t weight_count, char *filename) {
    int *dims = (int *) malloc(sizeof(int) * size);
    memcpy(dims, dims_array, sizeof(int) * size);
    float *weights = (float *) malloc(sizeof(float) * weight_count);
    struct Shape shape = {dims, size};
    params[idx] = (struct ParamInfo){shape, weights, filename};
}


// Sets up the ParamInfo fields of the params array
/**
 * @brief Initializes each element of the params array with the appropriate specification
 * Uses the above helper function
 */
static void init_params() {
    init_param(CONV10W_IDX, (int[]){16, 1, 3, 3, 3}, 5, 16 * 3 * 3 * 3, "conv_layer1.0.weight.bin");
    init_param(CONV10B_IDX, (int[]){16}, 1, 16, "conv_layer1.0.bias.bin");

    init_param(CONV12W_IDX, (int[]){16}, 1, 16, "conv_layer1.2.weight.bin");
    init_param(CONV12B_IDX, (int[]){16}, 1, 16, "conv_layer1.2.bias.bin");
    init_param(CONV12MEAN_IDX, (int[]){16}, 1, 16, "conv_layer1.2.running_mean.bin");
    init_param(CONV12VAR_IDX, (int[]){16}, 1, 16, "conv_layer1.2.running_var.bin");

    init_param(CONV20W_IDX, (int[]){16, 16, 3, 3, 3}, 5, 16 * 16 * 3 * 3 * 3, "conv_layer2.0.weight.bin");
    init_param(CONV20B_IDX, (int[]){16}, 1, 16, "conv_layer2.0.bias.bin");

    init_param(CONV22W_IDX, (int[]){16}, 1, 16, "conv_layer2.2.weight.bin");
    init_param(CONV22B_IDX, (int[]){16}, 1, 16, "conv_layer2.2.bias.bin");
    init_param(CONV22MEAN_IDX, (int[]){16}, 1, 16, "conv_layer2.2.running_mean.bin");
    init_param(CONV22VAR_IDX, (int[]){16}, 1, 16, "conv_layer2.2.running_var.bin");

    init_param(SCONV10W_IDX, (int[]){3200, 1, 5, 5}, 4, 3200 * 5 * 5, "sepconv1.0.weight.bin");
    init_param(SCONV10B_IDX, (int[]){3200}, 1, 3200, "sepconv1.0.bias.bin");

    init_param(SCONV12W_IDX, (int[]){320, 3200, 1, 1}, 4, 320 * 3200, "sepconv1.2.weight.bin");
    init_param(SCONV12B_IDX, (int[]){320}, 1, 320, "sepconv1.2.bias.bin");

    init_param(SCONV14W_IDX, (int[]){320}, 1, 320, "sepconv1.4.weight.bin");
    init_param(SCONV14B_IDX, (int[]){320}, 1, 320, "sepconv1.4.bias.bin");
    init_param(SCONV14MEAN_IDX, (int[]){320}, 1, 320, "sepconv1.4.running_mean.bin");
    init_param(SCONV14VAR_IDX, (int[]){320}, 1, 320, "sepconv1.4.running_var.bin");

    init_param(SCONV20W_IDX, (int[]){320, 1, 3, 3}, 4, 320 * 3 * 3, "sepconv2.0.weight.bin");
    init_param(SCONV20B_IDX, (int[]){320}, 1, 320, "sepconv2.0.bias.bin");

    init_param(SCONV22W_IDX, (int[]){256, 320, 1, 1}, 4, 256 * 320, "sepconv2.2.weight.bin");
    init_param(SCONV22B_IDX, (int[]){256}, 1, 256, "sepconv2.2.bias.bin");

    init_param(SCONV24W_IDX, (int[]){256}, 1, 256, "sepconv2.4.weight.bin");
    init_param(SCONV24B_IDX, (int[]){256}, 1, 256, "sepconv2.4.bias.bin");
    init_param(SCONV24MEAN_IDX, (int[]){256}, 1, 256, "sepconv2.4.running_mean.bin");
    init_param(SCONV24VAR_IDX, (int[]){256}, 1, 256, "sepconv2.4.running_var.bin");

    init_param(SCONV30W_IDX, (int[]){256, 1, 3, 3}, 4, 256 * 3 * 3, "sepconv3.0.weight.bin");
    init_param(SCONV30B_IDX, (int[]){256}, 1, 256, "sepconv3.0.bias.bin");

    init_param(SCONV32W_IDX, (int[]){256, 256, 1, 1}, 4, 256 * 256, "sepconv3.2.weight.bin");
    init_param(SCONV32B_IDX, (int[]){256}, 1, 256, "sepconv3.2.bias.bin");

    init_param(SCONV34W_IDX, (int[]){256}, 1, 256, "sepconv3.4.weight.bin");
    init_param(SCONV34B_IDX, (int[]){256}, 1, 256, "sepconv3.4.bias.bin");
    init_param(SCONV34MEAN_IDX, (int[]){256}, 1, 256, "sepconv3.4.running_mean.bin");
    init_param(SCONV34VAR_IDX, (int[]){256}, 1, 256, "sepconv3.4.running_var.bin");

    init_param(FC1W_IDX, (int[]){16, 256}, 2, 16 * 256, "fc1.weight.bin");
    init_param(FC1B_IDX, (int[]){16}, 1, 16, "fc1.bias.bin");
}

/**
 * @brief prints the weights in the params array so we can check against the weights from python
 */
void print_weights() {
    for (int i = 0; i < NUM_PARAMS; i++) {
        struct ParamInfo parameter = params[i];
        printf("Parameter name: %s\n", parameter.filename);
        int wgt_size = get_wgt_size(parameter.shape);
        for (int j = 0; j < wgt_size; j++) {
            printf("%f ", parameter.weights[j]);
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
    int size = get_wgt_size(shape); // num of entries in a normal batch
    if (batch_num == NUM_BATCHES - 1) {
        dims[0] = 5;
        /* shape = (struct Shape) {(int[]){5, 1, 200, 5, 5}, 5}; */
    }
    char *file_name = "test_data.bin";
    
    // Concatenate the path of our binary
    char *new_buf = (char *) malloc(strlen(BIN_PATH) + strlen(file_name) + 1);
    memcpy(new_buf, BIN_PATH, strlen(BIN_PATH));
    memcpy(new_buf + strlen(BIN_PATH), file_name, strlen(file_name) + 1);

    FILE *f = fopen(new_buf, "rb");

    // Sanity checking:
    fseek(f, 0, SEEK_END);
    int f_size = ftell(f);
    int exp_size = size * (NUM_BATCHES - 1) * sizeof(float) + (5 * 200 * 5 * 5 * sizeof(float));
    printf("f_size is %d, the other number is %d\n", f_size, exp_size);
    assert(f_size == exp_size);
    rewind(f);
    
    fseek(f, size * sizeof(float) * batch_num, SEEK_SET);
    float *data_buf = (float *) malloc(sizeof(float) * get_wgt_size(shape));
    fread(data_buf, sizeof(float), get_wgt_size(shape), f);
    fclose(f);
    
    data = (struct ParamInfo) {.shape = shape, .weights = data_buf, .filename = file_name};
    free(new_buf);
}

/**
 * @brief Does the full initialization. We don't want to expose the other functions out of this file
 */
void full_weight_init() {
    init_params();
    init_weights();
}
