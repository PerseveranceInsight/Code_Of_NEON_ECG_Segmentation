#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <arm_neon.h>

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
    uint32_t h_steps = 0, w_steps = 0, remain_h_num = 0;
    uint32_t w_offset = 0;
    float *p_weight = NULL, *p_sig1 = NULL, *p_sig2 = NULL, *p_feat = NULL;
    float32x2_t vec_weight;
    float32x4_t vec_in_feature1, vec_in_feature2;
    float32x4_t vec_out_feature;
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
    h_steps = p_conv_weight->ori_l / FP_PACK_SIZE_H;
    w_steps = p_sig_ctr->cur_out_pack_l / FP_PACK_SIZE_W;
    remain_h_num = p_conv_weight->ori_l - h_steps*FP_PACK_SIZE_H;
    w_offset = w_steps*FP_PACK_SIZE_W;
    ree_log(GEMM_LOG, "%s h_steps %d, w_steps %d, remain_h_num %d, w_offset %d", __func__, h_steps, w_steps, remain_h_num, w_offset);
    p_weight = p_conv_weight->ori_buf;
    p_sig1 = p_sig_ctr->col_buf;
    p_sig2 = p_sig1 + w_offset;
    p_feat = p_out_feature->ori_buf;
    for (uint32_t h_step_ind = 0; h_step_ind<h_steps; h_step_ind++)
    {
        vec_weight = vld1_f32(p_weight);
        p_feat = p_out_feature->ori_buf;
        for (uint32_t w_step_ind = 0; w_step_ind<w_steps; w_step_ind++)
        {
            vec_in_feature1 = vld1q_f32(p_sig1);
            vec_in_feature2 = vld1q_f32(p_sig2);
            vec_out_feature = vld1q_f32(p_feat);
            vec_out_feature = vmlaq_lane_f32(vec_out_feature, vec_in_feature1, vec_weight, 0);
            vec_out_feature = vmlaq_lane_f32(vec_out_feature, vec_in_feature2, vec_weight, 1);
            vst1q_f32(p_feat, vec_out_feature);
            p_sig1 += FP_PACK_SIZE_W;
            p_sig2 += FP_PACK_SIZE_W;
            p_feat += FP_PACK_SIZE_W;
        }
        p_weight += FP_PACK_SIZE_H;
        p_sig1 += w_offset;
        p_sig2 += w_offset;
    }

    p_sig1 = p_sig_ctr->col_buf;
    p_weight = p_conv_weight->ori_buf;
    p_sig1 += h_steps*FP_PACK_SIZE_H*w_offset;
    p_weight += h_steps*FP_PACK_SIZE_H;
    p_feat = p_out_feature->ori_buf;
    vec_weight = vld1_f32(p_weight);

    if (remain_h_num!=0)
    {
        for (uint32_t w_step_ind = 0; w_step_ind<w_steps; w_step_ind++)
        {
            vec_in_feature1 = vld1q_f32(p_sig1);
            vec_out_feature = vld1q_f32(p_feat);
            vec_out_feature = vmlaq_lane_f32(vec_out_feature, vec_in_feature1, vec_weight, 0); 
            vst1q_f32(p_feat, vec_out_feature);
            p_sig1 += FP_PACK_SIZE_W;
            p_feat += FP_PACK_SIZE_W;
        }
    }

EXIT_ECG_SEG_FP_GEMM:
    GEMM_FUNC_EXIT;
    return retval;
}

int32_t ecg_seg_fp_add_bias(mat_sig_t *p_out_feature,
                            float bias,
                            BOOL fused_relu)
{
    GEMM_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    uint32_t w_steps = 0, remain_w_num = 0;
    float *p_feat = NULL;
    float32x4_t vec_bias, vec_out_feature;
    ree_check_null_exit_retval(p_out_feature, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_FP_ADD_BIAS,
                               "%s occurs error due to p_out_feature is NULL", __func__);
    ree_check_null_exit_retval(p_out_feature->ori_buf, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_FP_ADD_BIAS,
                               "%s occurs error due to p_out_feature->ori_buf is NULL", __func__);
    ree_log(GEMM_LOG, "%s bias %03.3f", __func__, bias);
    w_steps = p_out_feature->out_l / FP_PACK_SIZE_W;
    remain_w_num = p_out_feature->out_l - w_steps*FP_PACK_SIZE_W;
    ree_log(GEMM_LOG, "%s w_steps %d, remian_w_num %d", __func__, w_steps, remain_w_num);
    p_feat = p_out_feature->ori_buf;
    if (bias != 0.0f)
    {
        vec_bias = vdupq_n_f32(bias);
        for (uint32_t w_ind = 0; w_ind<w_steps; w_ind++)
        {
            vec_out_feature = vld1q_f32(p_feat);
            vec_out_feature = vaddq_f32(vec_out_feature, vec_bias);
            vst1q_f32(p_feat, vec_out_feature);
            p_feat += FP_PACK_SIZE_W;
        }
        for (uint32_t w_ind = 0; w_ind<remain_w_num; w_ind++)
        {
            *(p_feat) += bias;
            p_feat++; 
        }
    }
EXIT_ECG_SEG_FP_ADD_BIAS:
    GEMM_FUNC_EXIT;
    return retval;
}
