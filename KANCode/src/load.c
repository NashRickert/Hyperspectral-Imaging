#include "load.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Placeholder

const int TABLE_SIZE = TBL_SIZE;

/**
 * @brief initializes and returns a model with the correct number of layers,
 * nodes, etc. Notably does not initialize the lookup tables yet
 * @params widths is the buffer of layer widths as is used for the KAN in python
 * @params len is the length of the aforementioned buffer
 */
// I think this is done, but obviously will need to check later
// Note: I think we're not properly initializing the nodes after this correctly (?)
struct model init_model(int *widths, int len) {
    struct layer *layers = (struct layer *) malloc(sizeof(struct layer) * len);
    if (!layers) {
        printf("Malloc failed, init_model");
        exit(EXIT_FAILURE);
    }
    struct model model = {.layers = layers, .len = len};
    for (int i = 0; i < len; i++) {
        struct node *nodes = (struct node *) malloc(sizeof(struct node) * widths[i]);
        if (!nodes) {
            printf("Malloc failed, init_model");
            exit(EXIT_FAILURE);
        }
        layers[i].len = widths[i];
        layers[i].idx = i;
        for (int j = 0; j < widths[i]; j++) {
            struct node *node = nodes + j;
            // allocate for adder tree if it isn't the input layer
            if (i == 0) {
                node->tree.inputs = NULL;
            } else {
                node->tree.inputs = (float *) malloc(sizeof(float) * widths[i - 1]);
                if (!node->tree.inputs) {
                    printf("Malloc failed, init_model");
                    exit(EXIT_FAILURE);
                }
                node->tree.len = widths[i - 1];
                node->tree.ptr = 0;
            }
            // for now, we assume that each layer is fully connected
            // for generalization, this might need adjustment later
            // again, we do this for all but the last layer
            if (i == len - 1) {
                // we are at the last layer, we have no act funcs
                // also no next layer, etc.
                node->funcs = NULL;
                node->targets = NULL;
                node->next_layer = NULL;
            } else {
                // assume fully connected, so every thing in the next layer is a target
                int targ_len = widths[i + 1];
                node->targets = (int *) malloc(sizeof(int) * targ_len);
                for (int k = 0; k < targ_len; k++) {
                    node->targets[k] = k;
                }
                node->funcs = (struct act_fun *) malloc(sizeof(struct act_fun) * targ_len);

                node->len = targ_len;

                node->next_layer = layers + i + 1;
            }
        }
        layers[i].nodes = nodes;
    }
    return model;
}


/**
 * @brief cleans up all the dynamic memory associated with the model
 */
void cleanup_model(struct model model) {
    return;
}


/**
 * Takes a tensor and array of indices appropriate for the shape of the tensor
 * Returns the corresponding index of the entry in a 1d representation of the tensor
 * Note that we must have len(idxs) = tensor.shape.len
 * Also must have that 0 <= idxs[i] < shape.dim[i]
 */ 
int get_idx(struct Tensor *tensor, int *idxs) {
    /* printf("%s\n", __func__); */
    int sum = 0;
    /* printf("Tensor shape length: %d\n", tensor->shape.len); */
    for (int i = 0; i < tensor->shape.len; i++) {
        /* printf("prefixes[i]: %d\n", tensor->prefixes[i]); */
        int temp;
        sum += tensor->prefixes[i] * idxs[i];
    }
    return sum;
}


/**
 * Multiplies the elements of the shape to get the len of the associated tensor
 */
static int get_size(struct Shape shape) {
    int size = 1;
    for (int i = 0; i < shape.len; i++) {
        size *= shape.dim[i];
    }
    return size;
}


/**
 * Does the computation of prefixes associated with a shape
 * Prefixes are useful for calculating indexes, thus it is useful to store them
 * As a part of our tensor
 */
static int *compute_prefixes(struct Shape shape) {
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

    return prefixes;
}

/**
 * Constructs a tensor based on the passed shape parameter
 * The tensor uses the parameter shape itself in its shape field, so one should not reuse
 * shapes to create multiple tensors, otherwise their dimension arrays will be shared
 * This is all possible because everything about a tensor is uniquely determined by its shape
 * (Except its data values which will need to be filled in by the caller)
 */
struct Tensor construct_tensor(struct Shape shape) {
    int *prefixes = compute_prefixes(shape);
    int length = get_size(shape);
    float *data = (float *) malloc(sizeof(float) * length);
    if (data == NULL) {
        printf("Failed to successfully malloc\n");
        exit(EXIT_FAILURE);
    }

    int *dim_copy = (int *) malloc(sizeof(int) * shape.len);
    if (dim_copy == NULL) {
        printf("Failed to successfully malloc\n");
        exit(EXIT_FAILURE);
    }
    memcpy(dim_copy, shape.dim, sizeof(int) * shape.len);

    struct Tensor tens = {
        .shape = {.len = shape.len, .dim = dim_copy},
        .data = data,
        .prefixes = prefixes,
        .len = length,
    };
    return tens;
}

void destroy_tensor(struct Tensor *data) {
    free(data->shape.dim);
    free(data->data);
    free(data->prefixes);
}

/**
 * @brief This function initializes all the lookup tables in a layer based on tensors passed from
 * python
 * @param tbl_vals: Of shape (TBL_SIZE, layer_len, next layer_len)
 * For tens[i,j,k], holds the ith table value of the act_func for the jth node on this layer going
 * to the kth node on the next layer
 * @param lkup_meta_info: Of shape (layer_len, 4). Holds the additional 4 pieces of meta info
 * For each nodes lookup tables: xmin, xmax, xdist, inv_xdist in that order.
 * Should be the same for every node
 * @param layer: The layer of our model we are targetting
 */
void fill_lkup_tables(struct Tensor *tbl_vals, struct Tensor *lkup_meta_info, struct layer *layer) {
    /* printf("%s\n", __func__); */
    assert(tbl_vals->shape.len == 3);
    assert(tbl_vals->shape.dim[0] == TBL_SIZE);
    assert(tbl_vals->shape.dim[1] == layer->len);
    assert(tbl_vals->shape.dim[2] == layer->nodes[0].next_layer->len);
    assert(lkup_meta_info->shape.len == 2);
    assert(lkup_meta_info->shape.dim[0] == layer->len);
    assert(lkup_meta_info->shape.dim[1] == 4);

    for (int i = 0; i < layer->len; i++) {
        struct node *node = layer->nodes + i;

        float xmin = lkup_meta_info->data[layer->len * 0 + i];
        float xmax = lkup_meta_info->data[layer->len * 1 + i];
        float xdist = lkup_meta_info->data[layer->len * 2 + i];
        float inv_xdist = lkup_meta_info->data[layer->len * 3 + i];
        /* printf("xmin, xmax, xdist, inv_xdist, %f, %f, %f, %f\n", xmin, xmax, xdist, inv_xdist); */

        assert(node->len == node->next_layer->len);
        for (int j = 0; j < node->len; j++) {
            /* printf("here\n"); */
            struct act_fun *func = node->funcs + j;
            /* printf("\n\n"); */
            for (int k = 0; k < TBL_SIZE; k++) {
                /* printf("here here\n"); */
                /* printf("k, i, j: %d, %d, %d\n", k, i, j); */
                /* printf("data ptr: %p\n", tbl_vals->data); */
                int idx = get_idx(tbl_vals, (int[]){k, i, j});
                /* printf("Idx: %d\n", idx); */
                float yval = tbl_vals->data[idx];
                /* printf("%f ", yval); */
                func->table.tbl[k] = yval;

                func->table.xmin = xmin;
                func->table.xmax = xmax;
                func->table.xdist = xdist;
                func->table.inv_xdist = inv_xdist;
            }
        }
    }
    /* printf("end of %s\n", __func__); */
}
