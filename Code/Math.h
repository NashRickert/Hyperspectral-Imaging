#pragma once

#include "load.h"

int get_idx(struct Tensor tensor, int *idxs);

void print_buf(int *buf, int len);
void fprint_buf(float *buf, int len);


struct Tensor BatchNorm3d(struct Parameter W, struct Parameter B, struct Parameter M, struct Parameter V, struct Tensor data);
struct Tensor ReLU(struct Tensor data);
struct Tensor Conv3d(struct Parameter weight_st, struct Parameter bias_st, struct Tensor in_data, int out_channels,
             int kernel_size, int padding, int stride, int dilation);
struct Tensor Conv2d(struct Parameter weight_st, struct Parameter bias_st, struct Tensor data,
                     int out_channels, int kernel_size, int padding, int stride, int dilation);
