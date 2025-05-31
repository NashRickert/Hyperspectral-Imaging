#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "CNN.h"
#include "Math.h"
#include "load_params.h"

int main() {
    full_weight_init();
    load_batch(0);
    struct Data data2 = Conv3d(params[0], params[1], data, 16, 3, 1, 1, 1);
    printf("Done with Conv3d\n");
    fprint_buf(data2.data, get_wgt_size(data2.shape));
    struct Data data3 = ReLU(data2);
    printf("Done with ReLU\n");
    fprint_buf(data3.data, get_wgt_size(data3.shape));
    struct Data data4 = BatchNorm3d(params[2], params[3], params[4], params[5], data3);
    printf("Done with BatchNorm3d\n");
    fprint_buf(data4.data, get_wgt_size(data4.shape));

    /* fprint_buf(data4.data, get_wgt_size(data4.shape)); */
    /* print_buf(data.shape.dim, data.shape.len); */
    /* printf("%d\n", get_wgt_size(data.shape)); */
    /* for (int i = 0; i < get_wgt_size(data.shape); i++) { */
    /*     printf("%f ", data.weights[i]); */
    /* } */

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
