#pragma once

#include "load.h"

int get_idx(struct Tensor *tensor, int *idxs);

void BatchNorm3d(struct Parameter W, struct Parameter B, struct Parameter M, struct Parameter V, struct Tensor *data);
void ReLU(struct Tensor *data);
void Conv3d(struct Parameter weight_st, struct Parameter bias_st, struct Tensor *data, int out_channels,
            int kernel_size, int padding, int stride, int dilation);
void Conv2d(struct Parameter weight_st, struct Parameter bias_st, struct Tensor *data,
            int out_channels, int kernel_size, int padding, int stride, int dilation);
void DepthwiseConv2d(struct Parameter weight_st, struct Parameter bias_st, struct Tensor *data,
                     int kernel_size, int padding, int stride, int dilation);
void BatchNorm2d(struct Parameter W, struct Parameter B, struct Parameter M, struct Parameter V, struct Tensor *data);
void AvgPool2d(struct Tensor *data, int kernel_size);
void Linear(int in_features, int out_features, struct Parameter weight, struct Parameter bias, struct Tensor *data);

void get_predictions(struct Tensor *data, int **retbuf, int *retlen);
