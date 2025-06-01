#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Math.h"
#include "load.h"

#define IN_SHAPE_LEN_3D 5
#define OUT_SHAPE_LEN_3D 5

#define IN_SHAPE_LEN_2D 4
#define OUT_SHAPE_LEN_2D 4

static struct Shape copy_shape(struct Shape shape) {
    int *new_dim = (int *) malloc(sizeof(int) * shape.len);
    memcpy(new_dim, shape.dim, shape.len * sizeof(int));
    struct Shape new_shape = {.dim = new_dim, .len = shape.len};
    return new_shape;
}
/**
 * @brief Does an non-inplace ReLU calculation for a buf of length len
 */
void ReLU(struct Tensor *data) {
    struct Shape shape = copy_shape(data->shape);
    struct Tensor out_tensor = construct_tensor(shape);
    for (int i = 0; i < out_tensor.len; i++) {
        out_tensor.data[i] = (data->data[i] > 0.0f) ? data->data[i] : 0.0f;
    }
    destroy_tensor(data);
    *data = out_tensor;
    /* return out_tensor; */
}

/**
 * @brief This function takes the shape of a tensor and an array of indices (of the same length as the shape tensor)
 * and returns the corresponding index of the entry in a 1d representation of the tensor
 */ 
int get_idx(struct Tensor *tensor, int *idxs) {
    int sum = 0;
    for (int i = 0; i < tensor->shape.len; i++) {
        sum += tensor->prefixes[i] * idxs[i];
    }
    return sum;
}

/**
 * @brief This function returns the shape of the output of Conv3d
 */
static struct Shape get_output_shape_Conv3d(struct Shape in_shape, int out_channels, int kernel_size, int padding, int stride, int dilation) {
    int D = in_shape.dim[2];
    int H = in_shape.dim[3];
    int W = in_shape.dim[4];
    int intermediate_calc = 2 * padding - (dilation * (kernel_size - 1)) - 1;
    int Dout = ((D + intermediate_calc) / stride + 1);
    int Hout = ((H + intermediate_calc) / stride + 1);
    int Wout = ((W + intermediate_calc) / stride + 1);

    int *dim = (int *) malloc(sizeof(int) * OUT_SHAPE_LEN_3D);
    struct Shape out_shape = {
        .dim = dim,
        .len = OUT_SHAPE_LEN_3D,
    };
    out_shape.dim[0] = in_shape.dim[0];
    out_shape.dim[1] = out_channels;
    out_shape.dim[2] = Dout;
    out_shape.dim[3] = Hout;
    out_shape.dim[4] = Wout;
    return out_shape;
}

/**
 * @brief This function performs the pytorch operation Conv3d (3d convolution)
 * returns a pointer to the output data
 * return parameters out_shape_ret returns the output shape of the array
 * Caller is repsonsible for freeing both the output data and output shape
 * Rest of the parameters should be self explanatory
 * Note that input_shape_len and output_shape_len are always known to be 5, so we use a macro
 */
void Conv3d(struct Parameter weight_st, struct Parameter bias_st, struct Tensor *data,
                     int out_channels, int kernel_size, int padding, int stride, int dilation) {
    struct Tensor out_tensor = construct_tensor(get_output_shape_Conv3d(data->shape, out_channels, kernel_size, padding, stride, dilation));

    float *wgt = weight_st.tensor.data;
    float *bias = bias_st.tensor.data;

    int *in_shape = data->shape.dim;
    int *out_shape = out_tensor.shape.dim;
    for (int n = 0; n < in_shape[0]; n++) {
        for (int c_out = 0; c_out < out_channels; c_out++) {
            for (int d = 0; d < out_shape[2]; d++) {
                for (int h = 0; h < out_shape[3]; h++) {
                    for (int w = 0; w < out_shape[4]; w++) {
                        float sum = bias[c_out];
                        for (int c_in = 0; c_in < in_shape[1]; c_in++) {
                            for (int kd = 0; kd < kernel_size; kd++) {
                                for (int kh = 0; kh < kernel_size; kh++) {
                                    for (int kw = 0; kw < kernel_size; kw++) {
                                        int id = d * stride + kd * dilation - padding;
                                        int ih = h * stride + kh * dilation - padding;
                                        int iw = w * stride + kw * dilation - padding;

                                        if (id >= 0 && id < in_shape[2] &&
                                            ih >= 0 && ih < in_shape[3] &&
                                            iw >= 0 && iw < in_shape[4]) {
                                            int idx1 = get_idx(data, (int[]){n, c_in, id, ih, iw});
                                            int idx2 = get_idx(&weight_st.tensor, (int[]){c_out,c_in,kd,kh,kw});
                                            sum += data->data[idx1] * wgt[idx2];
                                        }
                                    }
                                }
                            }
                        }
                        int idx = get_idx(&out_tensor, (int[]) {n,c_out,d,h,w});
                        out_tensor.data[idx] = sum;
                    }
                }
            }
        }
    }
    destroy_tensor(data);
    *data = out_tensor;
}


void BatchNorm3d(struct Parameter W, struct Parameter B, struct Parameter M, struct Parameter V, struct Tensor *data) {
    const float eps = 1.0f / 100000.0;
    struct Tensor out_tensor = construct_tensor(copy_shape(data->shape));
    int *in_dim = data->shape.dim;
    for (int n = 0; n < in_dim[0]; n++) {
        for (int c_in = 0; c_in < in_dim[1]; c_in++) {
            for (int d = 0; d < in_dim[2]; d++) {
                for (int h = 0; h < in_dim[3]; h++) {
                    for (int w = 0; w < in_dim[4]; w++) {
                        int idx = get_idx(data, (int[]) {n, c_in, d, h, w});
                        float x = data->data[idx];
                        float norm_x = (x - M.tensor.data[c_in]) / sqrt(V.tensor.data[c_in] + eps);
                        out_tensor.data[idx] = norm_x * W.tensor.data[c_in] + B.tensor.data[c_in];
                    }
                }
            }
        }
    }
    destroy_tensor(data);
    *data = out_tensor;
}


/**
 * @brief This function returns the shape of the output of Conv2d
 */
static struct Shape get_output_shape_Conv2d(struct Shape in_shape, int out_channels, int kernel_size, int padding, int stride, int dilation) {
    int H = in_shape.dim[2];
    int W = in_shape.dim[3];
    int intermediate_calc = 2 * padding - (dilation * (kernel_size - 1)) - 1;
    int Hout = ((H + intermediate_calc) / stride + 1);
    int Wout = ((W + intermediate_calc) / stride + 1);

    int *dim = (int *) malloc(sizeof(int) * OUT_SHAPE_LEN_2D);
    struct Shape out_shape = {
        .dim = dim,
        .len = OUT_SHAPE_LEN_2D,
    };
    out_shape.dim[0] = in_shape.dim[0];
    out_shape.dim[1] = out_channels;
    out_shape.dim[2] = Hout;
    out_shape.dim[3] = Wout;
    return out_shape;
}


void Conv2d(struct Parameter weight_st, struct Parameter bias_st, struct Tensor *data,
                     int out_channels, int kernel_size, int padding, int stride, int dilation) {
    struct Tensor out_tensor = construct_tensor(get_output_shape_Conv2d(data->shape, out_channels, kernel_size, padding, stride, dilation));

    float *wgt = weight_st.tensor.data;
    float *bias = bias_st.tensor.data;

    int *in_shape = data->shape.dim;
    int *out_shape = out_tensor.shape.dim;
    for (int n = 0; n < in_shape[0]; n++) {
        for (int c_out = 0; c_out < out_channels; c_out++) {
            for (int h = 0; h < out_shape[2]; h++) {
                for (int w = 0; w < out_shape[3]; w++) {
                    float sum = bias[c_out];
                    for (int c_in = 0; c_in < in_shape[1]; c_in++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int ih = h * stride + kh * dilation - padding;
                                int iw = w * stride + kw * dilation - padding;

                                if (ih >= 0 && ih < in_shape[2] &&
                                    iw >= 0 && iw < in_shape[3]) {
                                    int idx1 = get_idx(data, (int[]){n, c_in, ih, iw});
                                    int idx2 = get_idx(&weight_st.tensor, (int[]){c_out,c_in,kh,kw});
                                    sum += data->data[idx1] * wgt[idx2];
                                }
                            }
                        }
                    }
                    int idx = get_idx(&out_tensor, (int[]) {n,c_out,h,w});
                    out_tensor.data[idx] = sum;
                }
            }
        }
    }
    destroy_tensor(data);
    *data = out_tensor;
}

void DepthwiseConv2d(struct Parameter weight_st, struct Parameter bias_st, struct Tensor *data,
                              int kernel_size, int padding, int stride, int dilation) {
    int in_channels = data->shape.dim[1];
    // For depthwise: out_channels = in_channels
    struct Tensor out_tensor = construct_tensor(get_output_shape_Conv2d(data->shape, in_channels, kernel_size, padding, stride, dilation));
    float *wgt = weight_st.tensor.data;
    float *bias = bias_st.tensor.data;
    int *in_shape = data->shape.dim;
    int *out_shape = out_tensor.shape.dim;
    
    for (int n = 0; n < in_shape[0]; n++) {
        for (int c = 0; c < in_channels; c++) {  // Same channel index for input and output
            for (int h = 0; h < out_shape[2]; h++) {
                for (int w = 0; w < out_shape[3]; w++) {
                    float sum = bias[c];
                    // No inner channel loop - each channel processed independently
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int ih = h * stride + kh * dilation - padding;
                            int iw = w * stride + kw * dilation - padding;
                            if (ih >= 0 && ih < in_shape[2] &&
                                iw >= 0 && iw < in_shape[3]) {
                                int idx1 = get_idx(data, (int[]){n, c, ih, iw});
                                // Weight shape: [out_channels, 1, kh, kw] = [in_channels, 1, kh, kw]
                                int idx2 = get_idx(&weight_st.tensor, (int[]){c, 0, kh, kw});
                                sum += data->data[idx1] * wgt[idx2];
                            }
                        }
                    }
                    int idx = get_idx(&out_tensor, (int[]) {n, c, h, w});
                    out_tensor.data[idx] = sum;
                }
            }
        }
    }
    destroy_tensor(data);
    *data = out_tensor;
}

void BatchNorm2d(struct Parameter W, struct Parameter B, struct Parameter M, struct Parameter V, struct Tensor *data) {
    const float eps = 1.0f / 100000.0;
    struct Tensor out_tensor = construct_tensor(copy_shape(data->shape));
    int *in_dim = data->shape.dim;
    for (int n = 0; n < in_dim[0]; n++) {
        for (int c_in = 0; c_in < in_dim[1]; c_in++) {
            for (int h = 0; h < in_dim[2]; h++) {
                for (int w = 0; w < in_dim[3]; w++) {
                    int idx = get_idx(data, (int[]) {n, c_in, h, w});
                    float x = data->data[idx];
                    float norm_x = (x - M.tensor.data[c_in]) / sqrt(V.tensor.data[c_in] + eps);
                    out_tensor.data[idx] = norm_x * W.tensor.data[c_in] + B.tensor.data[c_in];
                }
            }
        }
    }
    destroy_tensor(data);
    *data = out_tensor;
}

void AvgPool2d(struct Tensor *data, int kernel_size) {
    int stride = kernel_size;
    int N = data->shape.dim[0];
    int C = data->shape.dim[1];
    int H = data->shape.dim[2];
    int W = data->shape.dim[3];
    int Hout = (H - kernel_size) / stride + 1;
    int Wout = (W - kernel_size) / stride + 1;

    struct Shape out_shape = copy_shape(data->shape);
    out_shape.dim[2] = Hout;
    out_shape.dim[3] = Wout;
    struct Tensor out_tensor = construct_tensor(out_shape);
    for (int h = 0; h < Hout; h++) {
        for (int w = 0; w < Wout; w++) {
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int idx = get_idx(data, (int[]){n,c,stride * h + kh, stride * w + kw});
                            sum += data->data[idx];
                        }
                    }
                    sum /= (kernel_size * kernel_size);
                    int idx = get_idx(&out_tensor, (int[]){n,c,h,w});
                    out_tensor.data[idx] = sum;
                }
            }
        }
    }
    destroy_tensor(data);
    *data = out_tensor;
}

void Linear(int in_features, int out_features, struct Parameter weight, struct Parameter bias, struct Tensor *data) {
    (void) in_features;
    struct Shape out_shape = copy_shape(data->shape);
    out_shape.dim[1] = out_features;
    struct Tensor out_tensor = construct_tensor(out_shape);
    for (int h = 0 ; h < out_tensor.shape.dim[0]; h++) {
        for (int w = 0; w < out_tensor.shape.dim[1]; w++) {
            float sum = 0.0f;
            for (int i = 0; i < data->shape.dim[1]; i++) {
                int idx1 = get_idx(data, (int[]){h,i});
                int idx2 = get_idx(&weight.tensor, (int[]){w,i});
                sum += data->data[idx1] * weight.tensor.data[idx2];
            }
            int idx = get_idx(&out_tensor, (int[]) {h,w});
            out_tensor.data[idx] = sum + bias.tensor.data[w];
        }
    }
    destroy_tensor(data);
    *data = out_tensor;
}
