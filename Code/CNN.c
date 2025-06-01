#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "CNN.h"
#include "Math.h"
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

int main() {
    struct Tensor data;
    full_weight_init();
    load_batch(0, &data);

    forward(&data);

    fprint_buf(data.data, data.len);
    return EXIT_SUCCESS;
}
