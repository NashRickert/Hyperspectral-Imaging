#include <stdio.h>
#include <stdlib.h>
#include "Math.h"
#include "load_params.h"

/**
 * @brief Does an inplace ReLU calculation for a buf of length len
 */
void ReLU(float *buf, int len) {
    for (int i = 0; i < len; i++) {
        buf[i] = (buf[i] > 0.0f) ? buf[i] : 0.0f;
    }
}

// Note: For repeated lookups it would be better to save the prefixes array as a global var or something
// OR: I could include it in param_info for the weights, and then have a single updating global var for the data as it passes through the network
// This is a todo for later, for now the simple implementation is fine
/**
 * @brief This function takes the shape of a tensor and an array of indices (of the same length as the shape tensor)
 * and returns the corresponding index of the entry in a 1d representation of the tensor
 */
int get_idx(struct Shape shape, int *idxs) {
    int dim_len = shape.len;
    int *dimensions = shape.dim;
    int *prefixes = (int *) malloc(sizeof(int) * dim_len);

    for (int i = dim_len - 1; i >= 0; i--) {
        if (i == dim_len - 1) {
            prefixes[i] = 1;
            continue;
        }
        prefixes[i] = prefixes[i + 1] * dimensions[i+1];
    }

    int sum = 0;
    for (int i = 0; i < dim_len; i++) {
        int recip = dim_len - i - 1;
        sum += prefixes[i] * idxs[i];
    }

    free(prefixes);
    return sum;
}

#define IN_SHAPE_LEN 5
#define OUT_SHAPE_LEN 5

static void print_buf(int *buf, int len) {
    for (int i = 0; i < len; i++) printf("%d ", buf[i]);
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
    out_shape.dim[2] = D;
    out_shape.dim[3] = H;
    out_shape.dim[4] = W;
    /* print_buf(out_shape, 5); */
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
float *Conv3d(struct ParamInfo weight_st, struct ParamInfo bias_st, float *in_data, struct Shape in_shape_st, int in_channels,
              int out_channels, int kernel_size, int padding, int stride, int dilation, struct Shape *out_shape_ret ) {
    int *in_shape = in_shape_st.dim;
    struct Shape out_shape_st = get_output_shape_Conv3d(in_shape_st, out_channels, kernel_size, padding, stride, dilation);
    int *out_shape = out_shape_st.dim;

    int out_len = 1;
    for (int i = 0; i < OUT_SHAPE_LEN; i++) out_len *= out_shape[i];
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
                                            sum += in_data[get_idx(in_shape_st, (int[]){n, c_in, id, ih, iw})]
                                                * wgt[get_idx(weight_st.shape, (int[]){c_out,c_in,kd,kh,kw})];
                                        }
                                    }
                                }
                            }
                        }
                        out_data[get_idx(out_shape_st, (int[]) {n,c_out,d,h,w})] = sum;
                    }
                }
            }
        }
    }
    *out_shape_ret = (struct Shape) {.dim = out_shape, .len = OUT_SHAPE_LEN};
    return out_data;
}
