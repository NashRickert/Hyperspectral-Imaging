#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "CNN.h"
#include "Math.h"
#include "load.h"

#define APPLY_LAYER(new_result) do { \
    struct Tensor temp = data; \
    data = (new_result); \
    destroy_tensor(temp); \
} while(0)

static int img_shape[4] = {1, 200, 5, 5};

void print_shape(struct Shape shape) {
    printf("Shape size: %d\n Shape dimensions:\n", shape.len);
    print_buf(shape.dim, shape.len);
}

// Every functino parameter should just become a pointer, but for now this is a fine substitute:
void forward(struct Tensor *data1) {
    struct Tensor data = *data1;
    APPLY_LAYER(Conv3d(params[CONV10W_IDX], params[CONV10B_IDX], data, 16, 3, 1, 1, 1));
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(BatchNorm3d(params[CONV12W_IDX], params[CONV12B_IDX], params[CONV12MEAN_IDX], params[CONV12VAR_IDX], data));

    printf("Done with round 1\n");
    print_shape(data.shape);

    APPLY_LAYER(Conv3d(params[CONV20W_IDX], params[CONV20B_IDX], data, 16, 3, 1, 1, 1));
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(BatchNorm3d(params[CONV22W_IDX], params[CONV22B_IDX], params[CONV22MEAN_IDX], params[CONV22VAR_IDX], data));

    printf("Done with round 2\n");
    print_shape(data.shape);

    // Do the reshaping here
    int x_shape_0 = data.shape.dim[0];
    free(data.shape.dim);

    data.shape.dim = (int *) malloc(sizeof(int) * 4);
    data.shape.len = 4;
    data.shape.dim[0] = x_shape_0;
    data.shape.dim[1] = img_shape[1] * 16;
    data.shape.dim[2] = img_shape[2];
    data.shape.dim[3] = img_shape[3];
    free(data.prefixes);
    data.prefixes = compute_prefixes(data.shape);

    APPLY_LAYER(DepthwiseConv2d(params[SCONV10W_IDX], params[SCONV10B_IDX], data, 5, 2, 1, 1));
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(Conv2d(params[SCONV12W_IDX], params[SCONV12B_IDX], data, 320, 1, 0, 1, 1));
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(BatchNorm2d(params[SCONV14W_IDX], params[SCONV14B_IDX], params[SCONV14MEAN_IDX], params[SCONV14VAR_IDX], data));

    printf("Done with round 3\n");
    print_shape(data.shape);
    
    // Small differences in the floating points are starting to appear now even though data mostly matches

    // Bad line here
    APPLY_LAYER(DepthwiseConv2d(params[SCONV20W_IDX], params[SCONV20B_IDX], data, 3, 1, 1, 1));
    /* assert(data.shape.dim[2] == 5); */
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(Conv2d(params[SCONV22W_IDX], params[SCONV22B_IDX], data, 256, 1, 0, 1, 1));
    /* assert(data.shape.dim[2] == 5); */
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(BatchNorm2d(params[SCONV24W_IDX], params[SCONV24B_IDX], params[SCONV24MEAN_IDX], params[SCONV24VAR_IDX], data));
    /* assert(data.shape.dim[2] == 5); */

    printf("Done with round 4\n");
    print_shape(data.shape);

    APPLY_LAYER(DepthwiseConv2d(params[SCONV30W_IDX], params[SCONV30B_IDX], data, 3, 1, 1, 1));
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(Conv2d(params[SCONV32W_IDX], params[SCONV32B_IDX], data, 256, 1, 0, 1, 1));
    APPLY_LAYER(ReLU(data));
    APPLY_LAYER(BatchNorm2d(params[SCONV34W_IDX], params[SCONV34B_IDX], params[SCONV34MEAN_IDX], params[SCONV34VAR_IDX], data));

    printf("Done with round 5\n");

    APPLY_LAYER(AvgPool2d(data, 5));
    print_shape(data.shape);

    // Do the reshaping here
    x_shape_0 = data.shape.dim[0];
    int x_shape_1 = data.shape.dim[1];
    free(data.shape.dim);

    data.shape.dim = (int *) malloc(sizeof(int) * 2);
    data.shape.len = 2;
    data.shape.dim[0] = x_shape_0;
    data.shape.dim[1] = x_shape_1;
    free(data.prefixes);
    data.prefixes = compute_prefixes(data.shape);

    APPLY_LAYER(Linear(256, 16, params[FC1W_IDX], params[FC1B_IDX], data));

    printf("Finally done\n");
    print_shape(data.shape);
    
    *data1 = data;
}

int main() {
    struct Tensor data;
    full_weight_init();
    load_batch(0, &data);

    forward(&data);

    fprint_buf(data.data, data.len);
    return EXIT_SUCCESS;
}
