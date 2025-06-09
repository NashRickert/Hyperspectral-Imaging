#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Math.h"
#include "load.h"
#include "counter.h"

#define IN_SHAPE_LEN_3D 5
#define OUT_SHAPE_LEN_3D 5

#define IN_SHAPE_LEN_2D 4
#define OUT_SHAPE_LEN_2D 4

// Each mathematical operation corresponds precisely with the pytorch function
// Of the same name for documentation version 2.7
// https://docs.pytorch.org/docs/2.7/

/**
 * Does a deep copy of the provided shape and returns it
 */
static struct Shape copy_shape(struct Shape shape) {
    int *new_dim = (int *) malloc(sizeof(int) * shape.len);
    memcpy(new_dim, shape.dim, shape.len * sizeof(int));
    struct Shape new_shape = {.dim = new_dim, .len = shape.len};
    return new_shape;
}
/**
 * Does a ReLU calculation on data in accordance with the pytorch specification
 */
void ReLU(struct Tensor *data) {
    printf("Inside of relu\n");
    struct Shape shape = copy_shape(data->shape);
    struct Tensor out_tensor = construct_tensor(shape);
    RELU_ACCUM(out_tensor.len);
    for (int i = 0; i < out_tensor.len; i++) {
        out_tensor.data[i] = (data->data[i] > 0.0f) ? data->data[i] : 0.0f;
    }
    destroy_tensor(data);
    *data = out_tensor;
    /* return out_tensor; */
}

/**
 * Takes a tensor and array of indices appropriate for the shape of the tensor
 * Returns the corresponding index of the entry in a 1d representation of the tensor
 * Note that we must have len(idxs) = tensor.shape.len
 * Also must have that 0 <= idxs[i] < shape.dim[i]
 */ 
int get_idx(struct Tensor *tensor, int *idxs) {
    int sum = 0;
    for (int i = 0; i < tensor->shape.len; i++) {
        int temp;
        IDX_MULT(tensor->prefixes[i], idxs[i], temp);
        IDX_ADD(sum, temp, sum);
        /* sum += tensor->prefixes[i] * idxs[i]; */
    }
    return sum;
}

/**
 * Returns the expected shape of the output of Conv3d
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
 * Performs the Conv3d operation on data in accordance with the pytorch specification
 */
void Conv3d(struct Parameter weight_st, struct Parameter bias_st, struct Tensor *data,
                     int out_channels, int kernel_size, int padding, int stride, int dilation) {
    struct Tensor out_tensor = construct_tensor(get_output_shape_Conv3d(data->shape, out_channels, kernel_size, padding, stride, dilation));
    printf("Inside of Conv3d\n");

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
                                        int temp1, temp2;
                                        int id, ih, iw;
                                        DATA_MULT(d, stride, temp1);
                                        DATA_MULT(kd, dilation, temp2);
                                        DATA_ADD(temp1, temp2, id);
                                        DATA_ADD(id, -padding, id);

                                        DATA_MULT(h, stride, temp1);
                                        DATA_MULT(kh, dilation, temp2);
                                        DATA_ADD(temp1, temp2, ih);
                                        DATA_ADD(ih, -padding, ih);

                                        DATA_MULT(w, stride, temp1);
                                        DATA_MULT(kw, dilation, temp2);
                                        DATA_ADD(temp1, temp2, iw);
                                        DATA_ADD(iw, -padding, iw);
                                        /* int id = d * stride + kd * dilation - padding; */
                                        /* int ih = h * stride + kh * dilation - padding; */
                                        /* int iw = w * stride + kw * dilation - padding; */

                                        if (id >= 0 && id < in_shape[2] &&
                                            ih >= 0 && ih < in_shape[3] &&
                                            iw >= 0 && iw < in_shape[4]) {
                                            int idx1 = get_idx(data, (int[]){n, c_in, id, ih, iw});
                                            int idx2 = get_idx(&weight_st.tensor, (int[]){c_out,c_in,kd,kh,kw});
                                            float temp;
                                            DATA_MULT(data->data[idx1], wgt[idx2], temp);
                                            DATA_ADD(sum, temp, sum);
                                            /* sum += data->data[idx1] * wgt[idx2]; */
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


/**
 * Performs the BatchNorm3d operation on data in accordance with the pytorch specification
 */
void BatchNorm3d(struct Parameter W, struct Parameter B, struct Parameter M, struct Parameter V, struct Tensor *data) {
    printf("Inside of BatchNorm3d\n");
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
                        float temp1;
                        float temp2 = 1 / sqrt(V.tensor.data[c_in] + eps);
                        DATA_ADD(x, -M.tensor.data[c_in], temp1);
                        DATA_MULT(temp1, temp2, temp1);
                        DATA_MULT(temp1, W.tensor.data[c_in], temp2);
                        DATA_ADD(temp2, B.tensor.data[c_in], out_tensor.data[idx]);
                        /* float norm_x = (x - M.tensor.data[c_in]) / sqrt(V.tensor.data[c_in] + eps); */
                        /* out_tensor.data[idx] = norm_x * W.tensor.data[c_in] + B.tensor.data[c_in]; */
                    }
                }
            }
        }
    }
    destroy_tensor(data);
    *data = out_tensor;
}

/**
 * Returns the expected shape of the output of Conv2d
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


/**
 * Performs the Conv2d operation on data in accordance with the pytorch specification
 */
void Conv2d(struct Parameter weight_st, struct Parameter bias_st, struct Tensor *data,
                     int out_channels, int kernel_size, int padding, int stride, int dilation) {
    printf("Inside of Conv2d\n");
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
                                int temp1, temp2;
                                int ih, iw;
                                DATA_MULT(h, stride, temp1);
                                DATA_MULT(kh, dilation, temp2);
                                DATA_ADD(temp1, temp2, ih);
                                DATA_ADD(ih, -padding, ih);

                                DATA_MULT(w, stride, temp1);
                                DATA_MULT(kw, dilation, temp2);
                                DATA_ADD(temp1, temp2, iw);
                                DATA_ADD(iw, -padding, iw);
                                /* int ih = h * stride + kh * dilation - padding; */
                                /* int iw = w * stride + kw * dilation - padding; */

                                if (ih >= 0 && ih < in_shape[2] &&
                                    iw >= 0 && iw < in_shape[3]) {
                                    int idx1 = get_idx(data, (int[]){n, c_in, ih, iw});
                                    int idx2 = get_idx(&weight_st.tensor, (int[]){c_out,c_in,kh,kw});
                                    float temp;
                                    DATA_MULT(data->data[idx1], wgt[idx2], temp);
                                    DATA_ADD(sum, temp, sum);
                                    /* sum += data->data[idx1] * wgt[idx2]; */
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

/**
 * Performs depthwise Conv2d on the data in accordance with the pytorch specification
 * This means that in_channels = groups = out_channels and is otherwise the same as Conv2d
 */
void DepthwiseConv2d(struct Parameter weight_st, struct Parameter bias_st, struct Tensor *data,
                              int kernel_size, int padding, int stride, int dilation) {
    printf("Inside of DepthwiseConv2d\n");
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
                            int temp1, temp2;
                            int ih, iw;
                            DATA_MULT(h, stride, temp1);
                            DATA_MULT(kh, dilation, temp2);
                            DATA_ADD(temp1, temp2, ih);
                            DATA_ADD(ih, -padding, ih);

                            DATA_MULT(w, stride, temp1);
                            DATA_MULT(kw, dilation, temp2);
                            DATA_ADD(temp1, temp2, iw);
                            DATA_ADD(iw, -padding, iw);
                            /* int ih = h * stride + kh * dilation - padding; */
                            /* int iw = w * stride + kw * dilation - padding; */
                            if (ih >= 0 && ih < in_shape[2] &&
                                iw >= 0 && iw < in_shape[3]) {
                                int idx1 = get_idx(data, (int[]){n, c, ih, iw});
                                // Weight shape: [out_channels, 1, kh, kw] = [in_channels, 1, kh, kw]
                                int idx2 = get_idx(&weight_st.tensor, (int[]){c, 0, kh, kw});
                                float temp;
                                DATA_MULT(data->data[idx1], wgt[idx2], temp);
                                DATA_ADD(sum, temp, sum);
                                /* sum += data->data[idx1] * wgt[idx2]; */
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

/**
 * Performs the BatchNorm2d operation on data in accordance with the pytorch specification
 */
void BatchNorm2d(struct Parameter W, struct Parameter B, struct Parameter M, struct Parameter V, struct Tensor *data) {
    printf("Inside of BatchNorm2d\n");
    const float eps = 1.0f / 100000.0;
    struct Tensor out_tensor = construct_tensor(copy_shape(data->shape));
    int *in_dim = data->shape.dim;
    for (int n = 0; n < in_dim[0]; n++) {
        for (int c_in = 0; c_in < in_dim[1]; c_in++) {
            for (int h = 0; h < in_dim[2]; h++) {
                for (int w = 0; w < in_dim[3]; w++) {
                    int idx = get_idx(data, (int[]) {n, c_in, h, w});
                    float x = data->data[idx];

                    float temp1;
                    float temp2 = 1 / sqrt(V.tensor.data[c_in] + eps);
                    DATA_ADD(x, -M.tensor.data[c_in], temp1);
                    DATA_MULT(temp1, temp2, temp1);
                    DATA_MULT(temp1, W.tensor.data[c_in], temp2);
                    DATA_ADD(temp2, B.tensor.data[c_in], out_tensor.data[idx]);
                    /* float norm_x = (x - M.tensor.data[c_in]) / sqrt(V.tensor.data[c_in] + eps); */
                    /* out_tensor.data[idx] = norm_x * W.tensor.data[c_in] + B.tensor.data[c_in]; */
                }
            }
        }
    }
    destroy_tensor(data);
    *data = out_tensor;
}

/**
 * Performs the AvgPool2d operation on data in accordance with the pytorch specification
 */
void AvgPool2d(struct Tensor *data, int kernel_size) {
    printf("Inside of AvgPool2d\n");
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
                            DATA_ADD(sum, data->data[idx], sum);
                            /* sum += data->data[idx]; */
                        }
                    }
                    float temp1 = (1.0f / (kernel_size * kernel_size));
                    DATA_MULT(temp1, sum, sum);

                    /* sum /= (kernel_size * kernel_size); */
                    int idx = get_idx(&out_tensor, (int[]){n,c,h,w});
                    out_tensor.data[idx] = sum;
                }
            }
        }
    }
    destroy_tensor(data);
    *data = out_tensor;
}

/**
 * Performs the Linear operation on data in accordance with the pytorch specification
 */
void Linear(int in_features, int out_features, struct Parameter weight, struct Parameter bias, struct Tensor *data) {
    printf("Inside of Linear\n");
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

                float temp;
                DATA_MULT(data->data[idx1], weight.tensor.data[idx2], temp);
                DATA_ADD(sum, temp, sum);
                /* sum += data->data[idx1] * weight.tensor.data[idx2]; */
            }
            int idx = get_idx(&out_tensor, (int[]) {h,w});
            DATA_ADD(sum, bias.tensor.data[w], out_tensor.data[idx]);
            /* out_tensor.data[idx] = sum + bias.tensor.data[w]; */
        }
    }
    destroy_tensor(data);
    *data = out_tensor;
}

/**
 * Takes a tensor which is the result of a forward pass of a batch of a data
 * Returns our predictions based on that data through output parameters
 * Applies log(softmax(data)) and then does argmax
 * Tensor returned is 1 dimensional, shape = batch_size = 128
 */
void get_predictions(struct Tensor *data, int **retbuf, int *retlen) {
    struct Shape in_shape = data->shape;
    int H = in_shape.dim[0];
    int W = in_shape.dim[1];
    assert(H == 128);
    assert(W == 16);

    // Used for storing output
    struct Tensor tensor = construct_tensor(copy_shape(data->shape));

    // Does the summation along each row for use in softmax
    float *denoms = (float *) malloc(sizeof(float) * H);
    for (int h = 0; h < H; h++) {
        float sum = 0.0f;
        for (int w = 0; w < W; w++) {
            float val = data->data[get_idx(data, (int[]) {h,w})];
            sum += expf(val);
        }
        denoms[h] = sum;
    }

    // Actually doing log softmax calculation
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            int idx = get_idx(data, (int[]) {h,w});
            float val = data->data[idx];
            tensor.data[idx] = logf(expf(val) / denoms[h]);
        }
    }
    free(denoms);

    // Does the argmax operation
    *retbuf = (int *) malloc(sizeof(int) * H);
    *retlen = H;
    for (int h = 0; h < H; h++) {
        int max_idx = 0;
        float max = tensor.data[get_idx(&tensor, (int[]){h,0})];
        for (int w = 0; w < W; w++) {
            float val = tensor.data[get_idx(&tensor, (int[]){h,w})];
            if (val > max) {
                max = val;
                max_idx = w;
            }
        }
        (*retbuf)[h] = max_idx;
    }

    destroy_tensor(&tensor);
} 
