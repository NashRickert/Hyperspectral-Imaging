#pragma once

#include <inttypes.h>

extern uint64_t data_add_accum_ctr;
extern uint64_t data_add_ctr;

extern uint64_t data_mult_accum_ctr;
extern uint64_t data_mult_ctr;

extern uint64_t idx_add_accum_ctr;
extern uint64_t idx_add_ctr;

extern uint64_t idx_mult_accum_ctr;
extern uint64_t idx_mult_ctr;

extern uint64_t relu_accum_ctr;
extern uint64_t relu_ctr;

#define RELU_ACCUM(a) \
    do { \
        relu_accum_ctr += (a); \
        relu_ctr += (a); \
    } while(0)


#define IDX_ADD(a, b, c) \
    do { \
        idx_add_accum_ctr++; \
        idx_add_ctr++; \
        (c) = (a) + (b); \
    } while(0)


#define IDX_MULT(a, b, c) \
    do { \
        idx_mult_accum_ctr++; \
        idx_mult_ctr++; \
        (c) = (a) * (b); \
    } while(0)


#define DATA_ADD(a, b, c) \
    do { \
        data_add_accum_ctr++; \
        data_add_ctr++; \
        (c) = (a) + (b); \
    } while(0)


#define DATA_MULT(a, b, c) \
    do { \
        data_mult_accum_ctr++; \
        data_mult_ctr++; \
        (c) = (a) * (b); \
    } while(0)
