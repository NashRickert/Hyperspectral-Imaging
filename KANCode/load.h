#include <stdbool.h>

#pragma once

#define NUM_LAYERS 5
#define TBL_SIZE 256

extern const int width[NUM_LAYERS];
extern const char *files[NUM_LAYERS];

struct adder_tree {
    float *inputs;    // Stores the values we accumulate from previous layers in the tree
    int len;          // len of inputs
    int ptr;          // the ptr for the final position of inputs that is actually used
                      // (it might not all be used if our layers aren't fully connected)
};

struct lkup_tbl {
    float tbl[TBL_SIZE];
    float xmin;              // x val associated with table[0]
    float xmax;              // x val associated with table[TBL_SIZE - 1]
    float xdist;             // dist between x values. Roughly (xmax - xmin) / TBL_SIZE
    float inv_xdist;         // the recipricol of xdist for division
};

struct act_fun {
    int *targets;         // The indexes of act_fun in the next layer that this act_fun passes vals to
                          // (model might not be fully connected)
    int tgt_len;          // len of targets
    bool fully_connected; // if true, ignore above fields and treat as if targets == all
    struct adder_tree;    // The adder tree storing inputs to this function
    struct lkup_tbl;      // The lookup table associated with this function
};

struct layer {
    struct act_fun *funcs;   // The act_funcs that compose this layer
    int len;                 // Len of above buffer
};

struct model {
    struct layer **layers;   // Pointer to buffer of layer pointers that point to the layers of the model
    int len;                 // Len of the above buffer
};





