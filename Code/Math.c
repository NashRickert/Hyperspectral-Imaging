#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Math.h"
#include "load_params.h"

static struct Shape copy_shape(struct Shape shape) {
    int *new_dim = (int *) malloc(sizeof(float) * shape.len);
    memcpy(new_dim, shape.dim, shape.len);
    struct Shape new_shape = {.dim = new_dim, .len = shape.len};
    return new_shape;
}
/**
 * @brief Does an non-inplace ReLU calculation for a buf of length len
 */
struct Data ReLU(struct Data data) {
    int len = get_wgt_size(data.shape);
    printf("ReLU len is %d\n", len);
    print_buf(data.shape.dim, data.shape.len);
    float *out_data_buf = (float *) malloc(sizeof(float) * len);
    for (int i = 0; i < len; i++) {
        out_data_buf[i] = (data.data[i] > 0.0f) ? data.data[i] : 0.0f;
    }
    struct Data out_data = {.shape = copy_shape(data.shape), .data = out_data_buf};
    return out_data;

}

// Note: For repeated lookups it would be better to save the prefixes array as a global var or something
// OR: I could include it in param_info for the weights, and then have a single updating global var for the data as it passes through the network
// This is a todo for later, for now the simple implementation is fine
/**
 * @brief This function takes the shape of a tensor and an array of indices (of the same length as the shape tensor)
 * and returns the corresponding index of the entry in a 1d representation of the tensor
 */
int get_idx(struct Shape shape, int *idxs) {
    // For now this is fine, but with cahced prefixes the other version will be faster
    int dim_len = shape.len;
    int *dimensions = shape.dim;
    int offset = 0;
    int stride = 1;

    for (int i = dim_len - 1; i >= 0; i--) {
        offset += idxs[i] * stride;
        stride *= dimensions[i];
    }
    return offset;
    /* int dim_len = shape.len; */
    /* int *dimensions = shape.dim; */
    /* int *prefixes = (int *) malloc(sizeof(int) * dim_len); */

    /* for (int i = dim_len - 1; i >= 0; i--) { */
    /*     if (i == dim_len - 1) { */
    /*         prefixes[i] = 1; */
    /*         continue; */
    /*     } */
    /*     prefixes[i] = prefixes[i + 1] * dimensions[i+1]; */
    /* } */

    /* int sum = 0; */
    /* for (int i = 0; i < dim_len; i++) { */
    /*     sum += prefixes[i] * idxs[i]; */
    /* } */

    /* free(prefixes); */
    /* return sum; */
}

#define IN_SHAPE_LEN 5
#define OUT_SHAPE_LEN 5

void print_buf(int *buf, int len) {
    for (int i = 0; i < len; i++) printf("%d ", buf[i]);
    printf("\n\n");
}

void fprint_buf(float *buf, int len) {
    for (int i = 0; i < len; i++) printf("%f ", buf[i]);
    printf("\n\n");
}

/**
 * @brief This function returns the shape of the output of Conv3d
 */
struct Shape get_output_shape_Conv3d(struct Shape in_shape, int out_channels, int kernel_size, int padding, int stride, int dilation) {
    /* int *out_shape = (int *) malloc(sizeof(int) * OUT_SHAPE_LEN); */
    int D = in_shape.dim[2];
    int H = in_shape.dim[3];
    int W = in_shape.dim[4];
    int intermediate_calc = 2 * padding - (dilation * (kernel_size - 1)) - 1;
    int Dout = ((D + intermediate_calc) / stride + 1);
    int Hout = ((H + intermediate_calc) / stride + 1);
    int Wout = ((W + intermediate_calc) / stride + 1);

    int *dim = (int *) malloc(sizeof(int) * OUT_SHAPE_LEN);
    struct Shape out_shape = {
        .dim = dim,
        .len = OUT_SHAPE_LEN,
    };
    out_shape.dim[0] = in_shape.dim[0];
    out_shape.dim[1] = out_channels;
    out_shape.dim[2] = Dout;
    out_shape.dim[3] = Hout;
    out_shape.dim[4] = Wout;
    /* print_buf(out_shape, 5); */
    return out_shape;
}

// Note to self: This function has become kinda a mess as I've messed with my structs, and should be tweaked a little for legibility
/**
 * @brief This function performs the pytorch operation Conv3d (3d convolution)
 * returns a pointer to the output data
 * return parameters out_shape_ret returns the output shape of the array
 * Caller is repsonsible for freeing both the output data and output shape
 * Rest of the parameters should be self explanatory
 * Note that input_shape_len and output_shape_len are always known to be 5, so we use a macro
 */
struct Data Conv3d(struct ParamInfo weight_st, struct ParamInfo bias_st, struct Data data, int out_channels,
              int kernel_size, int padding, int stride, int dilation) {
    struct Shape in_shape_st = data.shape;
    int *in_shape = data.shape.dim;
    struct Shape out_shape_st = get_output_shape_Conv3d(data.shape, out_channels, kernel_size, padding, stride, dilation);
    print_buf(out_shape_st.dim, out_shape_st.len);
    int out_len = get_wgt_size(out_shape_st);
    float *out_data = (float *) malloc(sizeof(float) * out_len);

    float *wgt = weight_st.weights;
    float *bias = bias_st.weights;

    for (int n = 0; n < in_shape[0]; n++) {
        for (int c_out = 0; c_out < out_channels; c_out++) {
            for (int d = 0; d < in_shape[2]; d++) {
                for (int h = 0; h < in_shape[3]; h++) {
                    for (int w = 0; w < in_shape[4]; w++) {
                        float sum = bias[c_out];
                        for (int c_in = 0; c_in < in_shape[1]; c_in++) {
                            for (int kd = 0; kd < kernel_size; kd++) {
                                for (int kh = 0; kh < kernel_size; kh++) {
                                    for (int kw = 0; kw < kernel_size; kw++) {
                                        int id = d + kd - 1;
                                        int ih = h + kh - 1;
                                        int iw = w + kw - 1;

                                        if (id >= 0 && id < in_shape[2] &&
                                            ih >= 0 && ih < in_shape[3] &&
                                            iw >= 0 && iw < in_shape[4]) {
                                            /* printf("No fault"); */
                                            sum += data.data[get_idx(in_shape_st, (int[]){n, c_in, id, ih, iw})]
                                                * wgt[get_idx(weight_st.shape, (int[]){c_out,c_in,kd,kh,kw})];
                                        }
                                    }
                                }
                            }
                        }
                        /* printf("No fault"); */
                        int idx = get_idx(out_shape_st, (int[]) {n,c_out,d,h,w});
                        /* assert(idx < out_len); */
                        /* printf("Our indices are %d %d %d %d %d\n", n, c_out, d, h, w); */
                        out_data[idx] = sum;
                        /* printf("No fault"); */
                    }
                }
            }
        }
    }
    struct Data out_data_st = {.shape = out_shape_st, .data = out_data};

    return out_data_st;
}


struct Data BatchNorm3d(struct ParamInfo W, struct ParamInfo B, struct ParamInfo M, struct ParamInfo V, struct Data data) {
    float *out_data_buf = (float *) malloc(sizeof(float) * get_wgt_size(data.shape));
    const float eps = 1.0f / 100000.0;
    int *in_dim = data.shape.dim;
    for (int n = 0; n < in_dim[0]; n++) {
        for (int c_in = 0; c_in < in_dim[1]; c_in++) {
            for (int d = 0; d < in_dim[2]; d++) {
                for (int h = 0; h < in_dim[3]; h++) {
                    for (int w = 0; w < in_dim[4]; w++) {
                        int idx = get_idx(data.shape, (int[]) {n, c_in, d, h, w});
                        float x = data.data[idx];
                        float norm_x = (x - M.weights[c_in]) / sqrt(V.weights[c_in] + eps);
                        out_data_buf[idx] = norm_x * W.weights[c_in] + B.weights[c_in];
                    }
                }
            }
        }
    }
    struct Data ret_data = {.shape = copy_shape(data.shape), .data = out_data_buf};
    return ret_data;
}
