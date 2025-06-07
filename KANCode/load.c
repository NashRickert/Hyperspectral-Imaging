#include "load.h"
#include <math.h>
#include <assert.h>
// Placeholder

const int width[NUM_LAYERS] = {200, 32, 32, 32, 16};

/**
 * @brief Looks up the output value for input x for some lkup_tbl table
 */
float lookup(float x, struct lkup_tbl *table) {
    // Note: assuming our training data resembles test, this should be fine
    // Because our grid has adjusted to contain all of the inputs to the function in training
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
 * @brief Implements an adder tree algorithm to sum values stored in the tree
float accumulate(struct adder_tree *tree) {
    
}
