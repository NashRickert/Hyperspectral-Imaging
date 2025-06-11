#include <stdbool.h>

#pragma once

#define TBL_SIZE 256

/* extern const int width[NUM_LAYERS]; */
/* extern const char *files[NUM_LAYERS]; */

struct adder_tree {
    float *inputs;    // Stores the values we accumulate from previous layers in the tree
    int len;          // len of inputs
    int ptr;          // the ptr for the final position of inputs that is actually used
                      // (it might not all be used if our layers aren't fully connected)
                      // I declare that ptr is exclusive (so we use up to ptr -1)
                      // thus ptr = 0 means no valid entries
};

struct lkup_tbl {
    float tbl[TBL_SIZE];
    float xmin;              // x val associated with table[0]
    float xmax;              // x val associated with table[TBL_SIZE - 1]
    float xdist;             // dist between x values. Roughly (xmax - xmin) / TBL_SIZE
    float inv_xdist;         // the recipricol of xdist for division
};

struct act_fun {
    /* int *targets;         // The indexes of act_fun in the next layer that this act_fun passes vals to */
    /*                       // (model might not be fully connected) */
    /* int tgt_len;          // len of targets */
    /* bool fully_connected; // if true, ignore above fields and treat as if targets == all */
    /* struct adder_tree tree;    // The adder tree storing inputs to this function */
    struct lkup_tbl table;      // The lookup table associated with this function
};

struct node {
    struct adder_tree tree;    // This stores the accumulated values in the node from the prev layer
    float val;                  // This stores the added up vals from the tree
    struct act_fun *funcs;      // This stores a list of activation functions to apply when propogating to the next layer
    int *targets;               // this holds the index of nodes in the next layer that we should propogate to
    int len;                    // This is the length of the previous 2 bufs (should be the same)
    struct layer *next_layer;   // This points to the next layer
};

// TODO: I was thinking about whether I need a special node or layer for the final layer to know that we should get our outpute?

struct layer {
    struct node *nodes;   // The act_funcs that compose this layer
    int len;                 // Len of above buffer
    int idx;              // Self indexing on the layer to help a bit with come computations
};


struct model {
    struct layer *layers;   // Pointer to buffer of layer pointers that point to the layers of the model
    int len;                 // Len of the above buffer
};


/* float accumulate(struct adder_tree *tree); */
/* float lookup(float x, struct lkup_tbl *table); */

struct model init_model(int *widths, int len);
