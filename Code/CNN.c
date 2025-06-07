#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CNN.h"
/* #include "Math.h" */
#include "load.h"
#include "layers.h"

void print_shape(struct Shape shape) {
    printf("Shape size: %d\n Shape dimensions:\n", shape.len);
    for (int i = 0; i < shape.len; i++) printf("%d ", shape.dim[i]);
    printf("\n\n");
}


void forward(struct Tensor *data) {
    conv_layer1(data);
    conv_layer2(data);
    reshape1(data);
    sepconv1(data);
    // Small floating point differences appearing
    sepconv2(data);
    sepconv3(data);
    average(data);
    reshape2(data);
    fc1(data);
}

void fprint_buf(float *buf, int len) {
    for (int i = 0; i < len; i++) printf("%f ", buf[i]);
    printf("\n\n");
}

/**
 * This function loops through out batch_x_data.bin files which hold the output
 * of each forward pass for each batch of data. We then do the final transformation to get
 * our predictions and compare that against the val_data.bin which holds the correct results
 * We return our overall validation accuracy
 */
float ver_acc() {
    int tot_size = 128 * (NUM_BATCHES);
    int *results = (int *) malloc(sizeof(int) * tot_size);
    int *buf;
    int len;
    // Loop reads batch data, does final transformation, and stores it inside of results buffer
    for (int i = 0; i < NUM_BATCHES; i++) {
        char filename[64];
        sprintf(filename, "batch_%d_data.bin", i);

        char *new_buf = (char *) malloc(strlen(BIN_PATH) + strlen(filename) + 1);
        memcpy(new_buf, BIN_PATH, strlen(BIN_PATH));
        memcpy(new_buf + strlen(BIN_PATH), filename, strlen(filename) + 1);

        FILE *f = fopen(new_buf, "rb");
        free(new_buf);

        assert(f);
        int *dimens = (int *) malloc(sizeof(int) * 2);
        dimens[0] = 128;
        dimens[1] = 16;
        struct Shape shape = {.dim = dimens, .len = 2};
        struct Tensor tensor = construct_tensor(shape);

        fseek(f, 0, SEEK_END);
        int size = ftell(f) / sizeof(float);
        assert(size == tensor.len);
        rewind(f);

        fread(tensor.data, sizeof(float), tensor.len, f);
        fclose(f);

        get_predictions(&tensor, &buf, &len);
        int start = i * 128;
        memcpy(results + start, buf, 128 * sizeof(int));
        destroy_tensor(&tensor);
    }

    // Opens and stores data from val_data.bin inside of validation buffer
    char* file_name = "val_data.bin";
    char *new_buf = (char *) malloc(strlen(BIN_PATH) + strlen(file_name) + 1);
    memcpy(new_buf, BIN_PATH, strlen(BIN_PATH));
    memcpy(new_buf + strlen(BIN_PATH), file_name, strlen(file_name) + 1);

    FILE *f = fopen(new_buf, "rb");
    free(new_buf);
    fseek(f, 0, SEEK_END);
    int size = ftell(f) / sizeof(int);
    assert(size == tot_size);
    rewind(f);

    int *validation = (int *) malloc(sizeof(int) * tot_size);
    fread(validation, sizeof(int), tot_size, f);
    fclose(f);
    
    // Does correctness calculations
    int correct = 0;
    for (int i = 0; i < tot_size; i++) {
        if (results[i] == validation[i]) {
            correct++;
        }
        printf("results[i]: %d, validation[i]: %d\n", results[i], validation[i]);
    }
    float acc = ((float) correct) / ((float) tot_size);
    printf("Correctness: %f", acc);
    free(validation);
    free(results);
    return acc;
}

int main() {

    ver_acc();
    
  
    return EXIT_SUCCESS;
}
