#include <stdlib.h>
#include "Math.h"
#include "load.h"

void conv_layer1(struct Tensor *data);
void conv_layer2(struct Tensor *data);
void sepconv1(struct Tensor *data);
void sepconv2(struct Tensor *data);
void sepconv3(struct Tensor *data);
void average(struct Tensor *data);
void fc1(struct Tensor *data);
void reshape1(struct Tensor *data);
void reshape2(struct Tensor *data);
