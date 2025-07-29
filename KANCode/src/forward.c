#include "load.h"
#include "forward.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define POW_OF_TWO(a) ((a) & ((a) - 1)) == 0

/**
 * @brief Looks up the output value for input x for some lkup_tbl table
 */
static float lookup(float x, struct lkup_tbl *table) {
    // Note: assuming our training data resembles test, this should be fine
    // Because our grid has adjusted to contain all of the inputs to the function in training
    // If a bunch of values are outside of these bounds, we might encounter problems/inaccuracy
    if (x <= table->xmin) {
        return table->tbl[0];
    }
    else if (x >= table->xmax) {
        return table->tbl[TBL_SIZE - 1];
    }
    float idxf = (x - table->xmin) * table->inv_xdist;
    // remainder is now the proportion that x is between idx and idx + 1
    float remainder = fmodf(idxf, 1);
    assert(remainder >= 0);

    int idx = (int) idxf;
    return table->tbl[idx] + ((table->tbl[idx + 1] - table->tbl[idx]) * remainder);
}


/**
 * @brief Accumulates on a buffer that is sized as a power of 2
 * end is exclusive, so end = 8 means we have entries from 0 to 7
 */
static float accum_buf(float *buf, int end) {
    assert(POW_OF_TWO(end));
    int gap = 1;
    while (end != 1) {
        assert(end > 1);
        for (int i = 0; i < end; i += (gap * 2)) {
            assert(IS_NUMBER(buf[i]));
            assert(IS_NUMBER(buf[i + gap]));
            buf[i] += buf[i + gap];
        }
        end -= gap;
        gap *= 2;
    }
    return buf[0];
}

/**
 * @brief Implements an adder tree algorithm to sum values stored in the tree
 * It does this by breaking the buffer into sizes that are a power of 2
 * and then summing over those using the accum_buf function
 */
static float accumulate(struct adder_tree *tree) {
    // need to have at least 1 valid entry
    assert(tree->ptr >= 1);
    // Need the buf to have enough entries
    assert(tree->len >= tree->ptr);
    int start = 0;
    int multiplier = 1;
    int ptr_cpy = tree->ptr;
    float result = 0.0f;
    int end;
    while (ptr_cpy != 0) {
        end = ptr_cpy & 0b1;
        end *= multiplier;
        if (end != 0) {
            result += accum_buf(tree->inputs + start, end);
        }
        ptr_cpy = ptr_cpy >> 1;
        multiplier = multiplier << 1;
        start += end;
    }
    return result;
}


/**
 * @brief We propogate from this layer to the next layer
 * For each node in the layer, first accumulates values. Then for each target,
 * gets output of related act_func and places it in the proper entry in the
 * adder_tree of the next node. Must not be the last layer
 */
static void propogate(struct layer *layer) {
    assert(layer->len != 0);
    assert(layer->nodes[0].next_layer != NULL);
    for (int i = 0; i < layer->len; i++) {
        struct node *node = &(layer->nodes[i]);
        if (layer->idx != 0) {
            node->val = accumulate(&node->tree);
            assert(IS_NUMBER(node->val));
        }
        float input = node->val;
        for (int j = 0; j < node->len; j++) {
            int target = node->targets[j];
            // Should probably use target here for fully-connected agnostic implementation
            // Note that the code should almost definitely be
            /* struct act_fun *func = &(node->funcs[j]); */
            /* struct node *targ_node = &(node->next_layer->nodes[target]); */
            /* struct adder_tree *targ_tree = &targ_node->tree; */
            // But it's 7/28 rn and I'm reviewing this and I don't want to change it and then have
            // to test. For fully connected networks it doesn't matter.
            struct act_fun *func = &(node->funcs[j]);
            struct node *targ_node = &(node->next_layer->nodes[j]);
            struct adder_tree *targ_tree = &targ_node->tree;
            

            float output = lookup(input, &(func->table));
            assert(IS_NUMBER(output));
#ifdef SCALE
            float diff = func->table.ymax - func->table.ymin;
            output = output * diff + func->table.ymin;
            assert(IS_NUMBER(output));
#endif
            targ_tree->inputs[targ_tree->ptr] = output;
            targ_tree->ptr++;
        }
    }
}

/**
 * @brief Accumulates the values of the nodes in a layer
 * and returns as a buffer through return parameters
 */
static void ret_node_vals(struct layer *layer, float **retbuf, int *retlen) {
    *retbuf = (float *) malloc(sizeof(float) * layer->len);
    *retlen = layer->len;
    for (int i = 0; i < layer->len; i++) {
        struct node *node = &(layer->nodes[i]);
        node->val = accumulate(&node->tree);
        (*retbuf)[i] = node->val;
    }
}


/**
 * @brief does a complete forward pass of the model,
 * returning the results through the return parameters
 * @TODO: Make it so that this model can accept multiple batches at once
 */
void forward(struct model *model, float *input, int len, float **retbuf, int *retlen) {
    clock_t in_clock = clock();
    assert(len == model->layers->len);
    // Initializes the input values for the first layer
    for (int i = 0; i < len; i++) {
        struct node *node = model->layers->nodes + i;
        node->val = input[i];
    }
    // Propogates through each layer of the model
    // Does not propogate the last layer. Those nodes now hold the output values
    for (int i = 0; i < model->len - 1; i++) {
        propogate(model->layers + i);
    }
    // Return the output values from the last layer
    ret_node_vals(model->layers + model->len - 1, retbuf, retlen);
    clock_t out_clock = clock();
    double elapsed_time = (double) (out_clock - in_clock) / CLOCKS_PER_SEC;
    printf("Forward elapsed time in seconds for a single sample: %f\n", elapsed_time);
}


