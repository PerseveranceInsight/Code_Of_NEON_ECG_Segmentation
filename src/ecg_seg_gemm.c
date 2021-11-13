#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arm_util.h"
#include "arm_typedef.h"
#include "ecg_seg_gemm.h"
#include "ecg_response_def.h"
#include "ecg_seg_sig2col.h"
#include "ecg_seg_signal.h"

int32_t ecg_seg_fp_gemm(mat_sig_t *p_conv_weight,
                        sig2col_ctr_t *p_sig_ctr,
                        mat_sig_t *p_out_feature)
{
    GEMM_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_conv_weight, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_FP_GEMM,
                               "%s occurs error due to p_conv_weight is NULL", __func__);
    ree_check_null_exit_retval(p_sig_ctr, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_FP_GEMM,
                               "%s occurs error due to p_sig_ctr is NULL", __func__);
    ree_check_null_exit_retval(p_out_feature, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_FP_GEMM,
                               "%s occurs error due to p_out_feature is NULL", __func__);
    ree_check_null_exit_retval(p_conv_weight->ori_buf, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_FP_GEMM,
                               "%s occurs error due to p_conv_weight->ori_buf is NULL", __func__);
    ree_check_null_exit_retval(p_sig_ctr->col_buf, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_FP_GEMM,
                               "%s occurs error due to p_sig_ctr->col_buf is NULL", __func__);
    ree_check_null_exit_retval(p_out_feature->ori_buf, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_FP_GEMM,
                               "%s occurs error due to p_out_feature->ori_buf is NULL", __func__);
    ree_log(GEMM_LOG, "%s p_conv_weight->ori_l %d p_sig_ctr->cur_k_l %d", __func__, 
                                                                          p_conv_weight->ori_l,
                                                                          p_sig_ctr->cur_k_l);
    ree_log(GEMM_LOG, "%s p_sig_ctr->cur_out_pack_l %d p_out_feature->pack_w %d", __func__,
                                                                                 p_sig_ctr->cur_out_pack_l,
                                                                                 p_out_feature->pack_w);
    ree_log(GEMM_LOG, "%s p_sig_ctr->cur_out_l %d p_out_feature->ori_l %d", __func__,
                                                                            p_sig_ctr->cur_out_l,
                                                                            p_out_feature->ori_l);
EXIT_ECG_SEG_FP_GEMM:
    GEMM_FUNC_EXIT;
    return retval;
}
