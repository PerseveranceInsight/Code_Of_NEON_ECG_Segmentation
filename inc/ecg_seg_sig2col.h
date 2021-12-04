#ifndef __ECG_SEG_SIG2COL_H__
#define __ECG_SEG_SIG2COL_H__
#include <stdint.h>
#include <limits.h>
#include "arm_typedef.h"
#include "arm_util.h"
#include "ecg_seg_matrix.h"
#include "ecg_response_def.h"
#include "ecg_seg_util.h"

#ifdef EN_SIG2COL_DBG
#define SIG2COL_FUNC_ENTRANCE                   FUNC_ENTRANCE_LOG
#define SIG2COL_FUNC_EXIT                       FUNC_EXIT_LOG
#define SIG2COL_PRINTF(x...)                    ree_printf(LOG_DEBUG, x)
#define SIG2COL_LOG                             LOG_DEBUG
#else
#define SIG2COL_FUNC_ENTRANCE                   do {} while (0)
#define SIG2COL_FUNC_EXIT                       do {} while (0)
#define SIG2COL_PRINTF(x...)                    do {} while (0)
#define SIG2COL_LOG                             LOG_VERBOSE
#endif

#define SIG2COL_INVALID_IND                     (INT_MIN)

typedef struct sig2col_ctr {
    BOOL inited;
    uint32_t cur_out_l;
    uint32_t cur_out_pack_l;
    uint32_t cur_k_l;
    uint32_t cur_ele_num;
    uint32_t max_out_l;
    uint32_t max_out_pack_l;
    uint32_t max_k_l;
    uint32_t max_ele_num;
    void *col_buf;
} sig2col_ctr_t;

int32_t sig2col_ctr_fp_constructor(uint32_t max_out_l, uint32_t max_k_l, sig2col_ctr_t **pp_ctr);
int32_t sig2col_mat_fp(sig2col_ctr_t *p_ctr, mat_sig_t *p_mat);
int32_t sig2col_mat_tranconv_fp(sig2col_ctr_t *p_ctr, mat_sig_t *p_mat, mat_sig_para_t *p_para);
void sig2col_printf_mat_fp(sig2col_ctr_t *p_ctr);
void sig2col_ctr_destructor(sig2col_ctr_t *p_ctr);

#endif
