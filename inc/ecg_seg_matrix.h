#ifndef __ECG_SEG_MATRIX_H__
#define __ECG_SEG_MATRIX_H__
#include <stdint.h>

#include "arm_util.h"
#include "arm_typedef.h"

#ifdef MATRIX_DBG
#define MATRIX_FUNC_ENTRANCE                    FUNC_ENTRANCE_LOG
#define MATRIX_FUNC_EXIT                        FUNC_EXIT_LOG
#define MATRIX_PRINTF(x...)                     ree_printf(LOG_DEBUG, x)
#define MATRIX_LOG                              LOG_DEBUG
#else
#define MATRIX_FUNC_ENTRANCE                    do {} while (0)
#define MATRIX_FUNC_EXIT                        do {} while (0)
#define MATRIX_PRINTF(x...)                     do {} while (0)
#define MATRIX_LOG                              LOG_VERBOSE
#endif

#define FP_PACK_SIZE_H                          (2)
#define FP_PACK_SIZE_W                          (4)
#define FP_PACK_ELE                             (16)

#define ELE_U8_SIZE                             sizeof(uint8_t)
#define ELE_U16_SIZE                            sizeof(uint16_t)
#define ELE_U32_SIZE                            sizeof(uint32_t)
#define ELE_FP_SIZE                             sizeof(float)

typedef enum in_file_type {
    IN_FILE_U8 = 0,
    IN_FILE_U16 = 1,
    IN_FILE_U32 = 2,
    IN_FILE_FP = 3,
} in_file_type_t;

typedef enum mat_type {
    MAT_U8 = 0,
    MAT_U16 = 1,
    MAT_U32 = 2,
    MAT_FP = 3,
} mat_type_t;

typedef enum pack_mat {
    PACK_MAT_ORI = 0,
    PACK_MAT_COL = 1,
} pack_mat_t;

typedef struct mat_sig_para {
    uint32_t ori_l;
    uint32_t k_l;
    uint32_t padding;
    uint32_t stride;
} mat_sig_para_t;

typedef struct mat_sig {
    uint32_t ori_l;
    uint32_t out_l;
    uint32_t col_h;
    uint32_t col_w;
    uint32_t pack_w_step;
    uint32_t pack_h;
    uint32_t pack_w;
    uint32_t pack_ele;
    uint32_t padding;
    uint32_t stride;
    uint32_t alloc_col;
    void *ori_buf;
} mat_sig_t;

int32_t mat_sig_constructor_fp_st(mat_sig_para_t *p_param,
                                  mat_sig_t *p_mat);

int32_t mat_sig_constructor_fp_fopen(mat_sig_para_t *p_param,
                                     mat_sig_t *p_mat, 
                                     char *path,
                                     BOOL kernel);

int32_t mat_sig_constructor_fp(mat_sig_para_t *p_param,
                               mat_sig_t *p_mat,
                               BOOL kernel);

void mat_sig_destructor(mat_sig_t *p_mat);

#endif
