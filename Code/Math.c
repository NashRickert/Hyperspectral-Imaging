#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Math.h"

// Does an inplace relu calculation for a buf of length len
void ReLU(float *buf, int len) {
    for (int i = 0; i < len; i++) {
        buf[i] = (buf[i] > 0.0f) ? buf[i] : 0.0f;
    }
}

// Note: For repeated lookups it would be better to save the prefixes array as a global var or something
// OR: I could include it in param_info for the weights, and then have a single updating global var for the data as it passes through the network
// This is a todo for later, for now the simple implementation is fine
// The function takes the shape of a tensor (dimensions), the length of the dimensions,
// And an array of indices (also of length dim_len)
// And returns the corresponding index of the desired entry in a 1d representation of the tensor
int get_idx(int *dimensions, int dim_len, int *idxs) {
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

/* int main() { */
/*     int arr[] = {10, 0, 50, 3, 2}; */
/*     int dim[] = {128, 1, 200, 5, 5}; */
/*     printf("Index is %d, should be %d\n", get_idx(dim, 5, arr), 51267); */
/*     /\* printf("%f", FMAX(10.0, 20.0)); *\/ */
/* } */
