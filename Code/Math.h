#pragma once

#include "load_params.h"
int get_idx(struct Shape shape, int *idxs);
void ReLU(float *buf, int len);
struct Shape get_output_shape(struct Shape in_shape, int out_channels, int kernel_size, int padding, int stride, int dilation);
