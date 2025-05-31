#pragma once

#include "load.h"
int get_idx(struct Shape shape, int *idxs);
struct Tensor ReLU(struct Tensor data);
struct Shape get_output_shape_Conv3d(struct Shape in_shape, int out_channels, int kernel_size, int padding, int stride, int dilation);
void print_buf(int *buf, int len);
void fprint_buf(float *buf, int len);
struct Tensor Conv3d(struct Parameter weight_st, struct Parameter bias_st, struct Tensor in_data, int out_channels,
             int kernel_size, int padding, int stride, int dilation);
struct Tensor BatchNorm3d(struct Parameter W, struct Parameter B, struct Parameter M, struct Parameter V, struct Tensor data);
