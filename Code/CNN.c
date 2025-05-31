#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "CNN.h"
#include "Math.h"
#include "load.h"

void print_shape(struct Shape shape) {
    printf("Shape size: %d\n Shape dimensions:\n", shape.len);
    print_buf(shape.dim, shape.len);
}
int main() {
    full_weight_init();
    load_batch(0);

    /* struct Tensor temp_data; */

    // Note that this current technique leaks data because of the fact that we don't free
    // The old g_data tensors
    g_data = Conv3d(params[CONV10W_IDX], params[CONV10B_IDX], g_data, 16, 3, 1, 1, 1);
    printf("Done with Conv3d\n");

    g_data = ReLU(g_data);
    printf("Done with ReLU\n");

    g_data = BatchNorm3d(params[CONV12W_IDX], params[CONV12B_IDX], params[CONV12MEAN_IDX], params[CONV12VAR_IDX], g_data);
    printf("Done with BatchNorm3d\n");

    fprint_buf(g_data.data, g_data.len);

    return EXIT_SUCCESS;
}
