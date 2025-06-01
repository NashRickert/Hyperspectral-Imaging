#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "CNN.h"
#include "Math.h"
#include "load.h"

#define APPLY_LAYER(new_result) do { \
    struct Tensor temp = g_data; \
    g_data = (new_result); \
    destroy_tensor(temp); \
} while(0)

int img_shape[4] = {1, 200, 5, 5};

void print_shape(struct Shape shape) {
    printf("Shape size: %d\n Shape dimensions:\n", shape.len);
    print_buf(shape.dim, shape.len);
}

int main() {
    full_weight_init();
    load_batch(0);

    struct Tensor temp_data;

    // Note that this current technique leaks data because of the fact that we don't free
    // The old g_data tensors

    // What is the point of these 3d convolutions when input channel is just 1?
    APPLY_LAYER(Conv3d(params[CONV10W_IDX], params[CONV10B_IDX], g_data, 16, 3, 1, 1, 1));
    APPLY_LAYER(ReLU(g_data));
    APPLY_LAYER(BatchNorm3d(params[CONV12W_IDX], params[CONV12B_IDX], params[CONV12MEAN_IDX], params[CONV12VAR_IDX], g_data));

    APPLY_LAYER(Conv3d(params[CONV20W_IDX], params[CONV20B_IDX], g_data, 16, 3, 1, 1, 1));
    APPLY_LAYER(ReLU(g_data));
    APPLY_LAYER(BatchNorm3d(params[CONV22W_IDX], params[CONV22B_IDX], params[CONV22MEAN_IDX], params[CONV22VAR_IDX], g_data));

    // Do the reshaping here
    int x_shape_0 = g_data.shape.dim[0];
    free(g_data.shape.dim);

    g_data.shape.dim = (int *) malloc(sizeof(int) * 4);
    g_data.shape.len = 4;
    g_data.shape.dim[0] = x_shape_0;
    g_data.shape.dim[1] = img_shape[1];
    g_data.shape.dim[2] = img_shape[2];
    g_data.shape.dim[3] = img_shape[3];

    
    
    

    fprint_buf(g_data.data, g_data.len);

    return EXIT_SUCCESS;
}
