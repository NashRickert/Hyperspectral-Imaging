#include "load.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
// Placeholder

/**
 * @brief initializes and returns a model with the correct number of layers,
 * nodes, etc. Notably does not initialize the lookup tables yet
 * @params widths is the buffer of layer widths as is used for the KAN in python
 * @params len is the length of the aforementioned buffer
 */
// I think this is done, but obviously will need to check later
struct model init_model(int *widths, int len) {
    struct layer *layers = (struct layer *) malloc(sizeof(struct layer) * len);
    struct model model = {.layers = layers, .len = len};
    for (int i = 0; i < len; i++) {
        struct node *nodes = (struct node *) malloc(sizeof(struct node) * widths[i]);
        layers[i].len = widths[i];
        layers[i].idx = i;
        for (int j = 0; j < widths[i]; j++) {
            struct node *node = nodes + j;
            // allocate for adder tree if it isn't the input layer
            if (i == 0) {
                node->tree.inputs = NULL;
            } else {
                node->tree.inputs = (float *) malloc(sizeof(float) * widths[i - 1]);
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
