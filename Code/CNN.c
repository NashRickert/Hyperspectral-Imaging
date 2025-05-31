#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "CNN.h"
#include "Math.h"
#include "load_params.h"

void print_shape(struct Shape shape) {
    printf("Shape size: %d\n Shape dimensions:\n", shape.len);
    print_buf(shape.dim, shape.len);
}
int main() {
    full_weight_init();
    load_batch(0);

    data = Conv3d(params[CONV10W_IDX], params[CONV10B_IDX], data, 16, 3, 1, 1, 1);
    printf("Done with Conv3d\n");

    data = ReLU(data);
    printf("Done with ReLU\n");

    data = BatchNorm3d(params[CONV12W_IDX], params[CONV12B_IDX], params[CONV12MEAN_IDX], params[CONV12VAR_IDX], data);
    printf("Done with BatchNorm3d\n");

    /* fprint_buf(data.data, get_wgt_size(data.shape)); */

    return EXIT_SUCCESS;
}
