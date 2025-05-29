#define NUM_PARAMS 38

#define CONV10W_IDX 0
#define CONV10B_IDX 1
#define CONV12W_IDX 2
#define CONV12B_IDX 3
#define CONV12MEAN_IDX 4
#define CONV12VAR_IDX 5
#define CONV20W_IDX 6
#define CONV20B_IDX 7
#define CONV22W_IDX 8
#define CONV22B_IDX 9
#define CONV22MEAN_IDX 10
#define CONV22VAR_IDX 11
#define SCONV10W_IDX 12
#define SCONV10B_IDX 13
#define SCONV12W_IDX 14
#define SCONV12B_IDX 15
#define SCONV14W_IDX 16
#define SCONV14B_IDX 17
#define SCONV14MEAN_IDX 18
#define SCONV14VAR_IDX 19
#define SCONV20W_IDX 20
#define SCONV20B_IDX 21
#define SCONV22W_IDX 22
#define SCONV22B_IDX 23
#define SCONV24W_IDX 24
#define SCONV24B_IDX 25
#define SCONV24MEAN_IDX 26
#define SCONV24VAR_IDX 27
#define SCONV30W_IDX 28
#define SCONV30B_IDX 29
#define SCONV32W_IDX 30
#define SCONV32B_IDX 31
#define SCONV34W_IDX 32
#define SCONV34B_IDX 33
#define SCONV34MEAN_IDX 34
#define SCONV34VAR_IDX 35
#define FC1W_IDX 36
#define FC1B_IDX 37

struct ParamInfo {
    int *dimensions;
    int dim_len;     // The length of the dimensions array
    float *weights;  // Pointer to the malloced weights
    char *filename;
};

extern struct ParamInfo params[NUM_PARAMS];


/* const int conv10W[] = {16, 1, 3, 3, 3}; */
/* const int conv10B[] = {16}; */

/* const int conv12W[] = {16}; */
/* const int conv12B[] = {16}; */

/* const int conv12mean[] = {16}; */
/* const int conv12var[] = {16}; */

/* const int conv20W[] = {16, 16, 3, 3, 3}; */
/* const int conv20B[] = {16}; */

/* const int conv22W[] = {16}; */
/* const int conv22B[] = {16}; */

/* const int conv22mean[] = {16}; */
/* const int conv22var[] = {16}; */

/* const int sconv10W[] = {3200, 1, 5, 5}; */
/* const int sconv10B[] = {3200}; */

/* const int sconv12W[] = {320, 3200, 1, 1}; */
/* const int sconv12B[] = {320}; */

/* const int sconv14W[] = {320}; */
/* const int sconv14B[] = {320}; */

/* const int sconv14mean[] = {320}; */
/* const int sconv14var[] = {320}; */

/* const int sconv20W[] = {320, 1, 3, 3}; */
/* const int sconv20B[] = {320}; */

/* const int sconv22W[] = {256, 320, 1, 1}; */
/* const int sconv22B[] = {256}; */

/* const int sconv24W[] = {256}; */
/* const int sconv24B[] = {256}; */

/* const int sconv24mean[] = {256}; */
/* const int sconv24var[] = {256}; */

/* const int sconv30W[] = {256, 1, 3, 3}; */
/* const int sconv30B[] = {256}; */

/* const int sconv32W[] = {256, 256, 1, 1}; */
/* const int sconv32B[] = {256}; */

/* const int sconv34W[] = {256}; */
/* const int sconv34B[] = {256}; */

/* const int sconv34mean[] = {256}; */
/* const int sconv34var[] = {256}; */

/* const int fc1W[] = {16, 256}; */
/* const int fc1B[] = {16}; */
