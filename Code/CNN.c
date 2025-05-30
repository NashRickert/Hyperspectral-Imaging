#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "CNN.h"
#include "Math.h"
#include "load_params.h"

int main(int argc, char **argv) {
    full_weight_init();
    load_batch(NUM_BATCHES - 1);
    print_buf(data.shape.dim, data.shape.len);
    /* printf("%d\n", get_wgt_size(data.shape)); */
    for (int i = 0; i < get_wgt_size(data.shape); i++) {
        printf("%f ", data.weights[i]);
    }

    /* init_params(); */
    /* init_weights(); */
    /* print_weights(); */
    /* struct ParamInfo p = params[CONV10W_IDX]; */
    /* int idx = get_idx(p.dimensions, p.dim_len, (int []){13, 0, 1, 2, 0}); */
    /* float val = p.weights[idx]; */
    /* printf("%f", val); */
    /* printf(params[CONV10W_IDX]) */

    /* struct Shape in_shape = {(int[]) {128,1,200,5,5}, 5}; */
    /* struct Shape out_shape = get_output_shape_Conv3d(in_shape, 16, 3, 1, 1, 1); */
    /* print_buf(out_shape.dim, out_shape.len); */
    return EXIT_SUCCESS;
}
