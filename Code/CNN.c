#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "CNN.h"
#include "load_params.h"

int main(int argc, char **argv) {
    init_params();
    init_weights();
    print_weights();
    return EXIT_SUCCESS;
}
