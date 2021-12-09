#include <stdlib.h>
#include <string.h>

#include "arm_typedef.h"
#include "arm_util.h"
#include "ecg_seg_matrix.h"
#include "ecg_seg_sig2col.h"
#include "ecg_response_def.h"
#include "ecg_seg_util.h"

static void print_sig2col_ctr_param(sig2col_ctr_t *p_ctr)
{
    SIG2COL_FUNC_ENTRANCE;
    ree_check_null_exit(p_ctr, EXIT_PRINT_SIG2COL_CTR_PARAM, "%s occurs error due to p_ctr is NULL", __func__);
    SIG2COL_PRINTF("%s p_ctr->cur_out_l %d\n", __func__, p_ctr->cur_out_l);
    SIG2COL_PRINTF("%s p_ctr->cur_out_pack_l %d\n", __func__, p_ctr->cur_out_pack_l);
    SIG2COL_PRINTF("%s p_ctr->cur_k_l %d\n", __func__, p_ctr->cur_k_l);
    SIG2COL_PRINTF("%s p_ctr->cur_ele_num %d\n", __func__, p_ctr->cur_ele_num);
    SIG2COL_PRINTF("%s p_ctr->max_out_l %d\n", __func__, p_ctr->max_out_l);
    SIG2COL_PRINTF("%s p_ctr->max_out_pack_l %d\n", __func__, p_ctr->max_out_pack_l);
    SIG2COL_PRINTF("%s p_ctr->max_k_l %d\n", __func__, p_ctr->max_k_l);
    SIG2COL_PRINTF("%s p_ctr->max_ele_num %d\n", __func__, p_ctr->max_ele_num);
EXIT_PRINT_SIG2COL_CTR_PARAM:
    SIG2COL_FUNC_EXIT;
}

static inline float sig2col_get_pixel_fp(uint32_t sig_ind_w_padding, mat_sig_t *p_mat)
{
    SIG2COL_FUNC_ENTRANCE;
    float feature = 0.0f;
    float *p_buf = (float*)p_mat->ori_buf;
    int32_t sig_ind_wo_padding = sig_ind_w_padding - p_mat->padding;
    if ((sig_ind_wo_padding >= 0) && (sig_ind_wo_padding < p_mat->ori_l))
    {
        ree_log(SIG2COL_LOG, "sig_ind_wo_padding %d %d", sig_ind_w_padding, sig_ind_wo_padding);
        feature = *(p_buf+sig_ind_wo_padding);
    }
    ree_log(SIG2COL_LOG, "%04.4f\n", feature);
    SIG2COL_FUNC_EXIT;
    return feature;
}

static inline float sig2col_get_tranconv_pixel_fp(uint32_t col_h_ind, uint32_t col_w_ind, mat_sig_t *p_mat, mat_sig_tran_conv_para_t *p_para)
{
    SIG2COL_FUNC_ENTRANCE;
    float feature = 0.0f;
    uint32_t col_w_ind_w_padding = col_w_ind + p_para->padding;
    uint32_t feat_ind = (col_w_ind_w_padding - col_h_ind)/p_para->stride;
    float *p_buf = p_mat->ori_buf;
    if (((col_w_ind_w_padding+col_h_ind)%p_para->stride == 0) && (col_w_ind_w_padding >= col_h_ind) && (feat_ind < p_mat->ori_l))
    {
        ree_log(SIG2COL_LOG, "%s col_w_ind_w_padding %d col_h_ind %d col_w_ind %d feat_ind %d", __func__,
                                                                                                col_w_ind_w_padding,
                                                                                                col_h_ind,
                                                                                                col_w_ind,
                                                                                                feat_ind);
        feature = p_buf[feat_ind];
    }
    SIG2COL_FUNC_EXIT;
    return feature;
}

static inline float sig2col_get_deconv_pixel_fp(uint32_t sig_ind_w_padding, mat_sig_t *p_mat, mat_decoder_conv_para_t *p_para)
{
    SIG2COL_FUNC_ENTRANCE;
    float feature = 0.0f;
    float *p_buf = (float*)p_mat->ori_buf;
    uint32_t sig_ind_wo_padding = sig_ind_w_padding - p_para->padding;
    if ((sig_ind_wo_padding >= 0) && (sig_ind_wo_padding < p_mat->ori_l))
    {
        ree_log(SIG2COL_LOG, "sig_ind_wo_padding %d %d", sig_ind_w_padding, sig_ind_wo_padding);
        feature = *(p_buf+sig_ind_wo_padding);
    }
    SIG2COL_FUNC_EXIT;
    return feature;
}

void sig2col_printf_mat_fp(sig2col_ctr_t *p_ctr)
{
    SIG2COL_FUNC_ENTRANCE;
    float *p_col_buf = p_ctr->col_buf; 
    for (uint32_t col_h_ind = 0; col_h_ind<p_ctr->cur_k_l; col_h_ind++)
    {
        for (uint32_t col_w_ind = 0; col_w_ind<p_ctr->cur_out_pack_l; col_w_ind++)
        {
            SIG2COL_PRINTF("%04.4f ", *p_col_buf);
            p_col_buf++;
        }
        SIG2COL_PRINTF("\n");
    }

    SIG2COL_FUNC_EXIT;
}

int32_t sig2col_ctr_fp_constructor(uint32_t max_out_l, uint32_t max_k_l, sig2col_ctr_t **pp_ctr)
{
    SIG2COL_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    uint32_t ele_num = 0;
    ree_check_null_exit_retval(pp_ctr, retval, ECG_SEG_INVALID_PARAM, EXIT_SIG2COL_CTR_CONTRUCTOR,
                              "%s occurs error due to pp_ctr is NULL", __func__);
    if (!(*pp_ctr))
    {
        ree_log(SIG2COL_LOG, "%s prepares to allocate *pp_ctr", __func__);
        *pp_ctr = ree_malloc(sizeof(sig2col_ctr_t));
        ree_check_null_exit_retval((*pp_ctr), retval, ECG_SEG_ALLOC_FAILED, EXIT_SIG2COL_CTR_CONTRUCTOR,
                                   "%s occurs error due to pp_ctr is NULL", __func__);
    } else
    {
        ree_log(SIG2COL_LOG, "%s *pp_ctr has already been allocated", __func__);
    }
    ree_set(*pp_ctr, 0, sizeof(sig2col_ctr_t));
    (*pp_ctr)->max_out_l = max_out_l;
    (*pp_ctr)->max_out_pack_l = (max_out_l / FP_PACK_SIZE_W + 1) * FP_PACK_SIZE_W;
    (*pp_ctr)->max_k_l = max_k_l;
    (*pp_ctr)->max_ele_num = ((*pp_ctr)->max_out_pack_l)*((*pp_ctr)->max_k_l);
    print_sig2col_ctr_param(*pp_ctr);
    ele_num =  (*pp_ctr)->max_ele_num;
    ree_log(SIG2COL_LOG, "%s prepares to allocate %d floating point buffer", __func__, ele_num);
    (*pp_ctr)->col_buf = ree_malloc(ele_num*ELE_FP_SIZE);
    ree_check_null_exit_retval((*pp_ctr)->col_buf, retval, ECG_SEG_ALLOC_FAILED, EXIT_SIG2COL_CTR_CONTRUCTOR,
                               "%s occurs error due to *pp_ctr->col_buf is NULL", __func__);
    ree_set((*pp_ctr)->col_buf, 0, ele_num*ELE_FP_SIZE);
    (*pp_ctr)->inited = TRUE;
EXIT_SIG2COL_CTR_CONTRUCTOR:
    SIG2COL_FUNC_EXIT;
    return retval;
}

int32_t sig2col_mat_fp(sig2col_ctr_t *p_ctr, mat_sig_t *p_mat)
{
    SIG2COL_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    uint32_t ele_num = 0, sig_ind_w_padding = 0;
    float feature = 0.0f;
    float *p_col_buf = NULL;
    ree_check_null_exit_retval(p_mat, retval, ECG_SEG_INVALID_PARAM, EXIT_SIG2COL_MAT_FP,
                               "%s occurs error due to p_mat is NULL", __func__);
    ree_check_null_exit_retval(p_ctr, retval, ECG_SEG_INVALID_PARAM, EXIT_SIG2COL_MAT_FP,
                               "%s occurs error due to p_ctr is NULL", __func__);
    ree_check_null_exit_retval(p_ctr->col_buf, retval, ECG_SEG_INVALID_PARAM, EXIT_SIG2COL_MAT_FP,
                               "%s occurs error due to p_ctr->col_buf is NULL", __func__);
    ree_check_true_exit_retval((!p_ctr->inited), retval, ECG_SEG_INVALID_PARAM, EXIT_SIG2COL_MAT_FP,
                               "%s occurs error due to p_ctr->inited is FALSE", __func__);
    p_ctr->cur_k_l = p_mat->col_h;
    p_ctr->cur_out_l = p_mat->col_w;
    p_ctr->cur_out_pack_l = p_mat->pack_w;
    p_ctr->cur_ele_num = p_ctr->cur_k_l * p_ctr->cur_out_pack_l;
    print_sig2col_ctr_param(p_ctr);
    ele_num = p_ctr->cur_ele_num;
    p_col_buf = p_ctr->col_buf;
    ree_set(p_col_buf, 0, ele_num*ELE_FP_SIZE);
    
    for (uint32_t col_h_ind = 0; col_h_ind<p_ctr->cur_k_l; col_h_ind++)
    {
        sig_ind_w_padding = col_h_ind;
        for (uint32_t col_w_ind = 0; col_w_ind<p_ctr->cur_out_pack_l; col_w_ind++)
        {
            if (col_w_ind<p_ctr->cur_out_l)
            {
                feature = sig2col_get_pixel_fp(sig_ind_w_padding, p_mat);
            } else
            {
                feature = 0.0f;
            }
            *p_col_buf = feature;
            sig_ind_w_padding += p_mat->stride;
            p_col_buf++;
        }
    }

EXIT_SIG2COL_MAT_FP:
    SIG2COL_FUNC_EXIT;
    return retval;
}

int32_t sig2col_mat_tranconv_fp(sig2col_ctr_t *p_ctr, mat_sig_t *p_mat, mat_sig_tran_conv_para_t *p_para)
{
    SIG2COL_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    uint32_t ele_num = 0;
    float feature = 0.0f;
    float *p_col_buf = NULL;
    ree_check_null_exit_retval(p_ctr, retval, ECG_SEG_INVALID_PARAM, EXIT_SIG2COL_MAT_TRANCONV_FP,
                               "%s occurs error due to p_ctr is NULL", __func__);
    ree_check_null_exit_retval(p_mat, retval, ECG_SEG_INVALID_PARAM, EXIT_SIG2COL_MAT_TRANCONV_FP,
                               "%s occurs error due to p_mat is NULL", __func__);
    ree_check_null_exit_retval(p_para, retval, ECG_SEG_INVALID_PARAM, EXIT_SIG2COL_MAT_TRANCONV_FP,
                               "%s occurs error due to p_para is NULL", __func__);
    ree_check_null_exit_retval(p_ctr->col_buf, retval, ECG_SEG_INVALID_PARAM, EXIT_SIG2COL_MAT_TRANCONV_FP,
                               "%s occurs error due to p_ctr->col_buf is NULL", __func__);
    ree_check_true_exit_retval((!p_ctr->inited), retval, ECG_SEG_INVALID_PARAM, EXIT_SIG2COL_MAT_TRANCONV_FP,
                               "%s occurs error due to p_ctr->inited is FALSE", __func__);
    print_mat_sig_tran_conv_para(p_para);
    p_ctr->cur_k_l = p_para->col_h;
    p_ctr->cur_out_l = p_para->col_w;
    p_ctr->cur_out_pack_l = p_para->pack_w;
    p_ctr->cur_ele_num = p_ctr->cur_k_l * p_ctr->cur_out_pack_l;
    print_sig2col_ctr_param(p_ctr);
    ele_num = p_ctr->cur_ele_num;
    p_col_buf = p_ctr->col_buf;
    ree_set(p_col_buf, 0, ele_num*ELE_FP_SIZE);

    for (uint32_t col_h_ind = 0; col_h_ind<p_ctr->cur_k_l; col_h_ind++)
    {
        for (uint32_t col_w_ind = 0; col_w_ind<p_ctr->cur_out_pack_l; col_w_ind++)
        {
            if (col_w_ind < p_para->col_w)
            {
                feature = sig2col_get_tranconv_pixel_fp(col_h_ind, col_w_ind, p_mat, p_para);
            } else
            {
                feature = 0.0f;
            }
            *p_col_buf = feature;
            p_col_buf++;
        }
    }
EXIT_SIG2COL_MAT_TRANCONV_FP:
    SIG2COL_FUNC_EXIT;
    return retval;
}

int32_t sig2col_mat_decoder_mat_fp(sig2col_ctr_t *p_ctr, mat_sig_t *p_mat, mat_decoder_conv_para_t *p_para)
{
    SIG2COL_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    uint32_t ele_num = 0, sig_ind_w_padding = 0;
    float feature = 0.0f;
    float *p_col_buf = NULL;
    ree_check_null_exit_retval(p_ctr, retval, ECG_SEG_INVALID_PARAM, EXIT_SIG2COL_MAT_DECODER_MAT_FP,
                               "%s occurs error due to p_ctr is NULL", __func__);
    ree_check_null_exit_retval(p_mat, retval, ECG_SEG_INVALID_PARAM, EXIT_SIG2COL_MAT_DECODER_MAT_FP,
                               "%s occurs error due to p_mat is NULL", __func__);
    ree_check_null_exit_retval(p_para, retval, ECG_SEG_INVALID_PARAM, EXIT_SIG2COL_MAT_DECODER_MAT_FP,
                               "%s occurs error due to p_para is NULL", __func__);
    ree_check_null_exit_retval(p_ctr->col_buf, retval, ECG_SEG_INVALID_PARAM, EXIT_SIG2COL_MAT_DECODER_MAT_FP,
                               "%s occurs error due to p_ctr->col_buf is NULL", __func__);
    ree_check_true_exit_retval((!p_ctr->inited), retval, ECG_SEG_INVALID_PARAM, EXIT_SIG2COL_MAT_DECODER_MAT_FP,
                               "%s occurs error due to p_ctr->inited is FALSE", __func__);
    p_ctr->cur_k_l = p_para->col_h;
    p_ctr->cur_out_l = p_para->col_w;
    p_ctr->cur_out_pack_l = p_para->pack_w;
    p_ctr->cur_ele_num = p_ctr->cur_k_l * p_ctr->cur_out_pack_l;
    ele_num = p_ctr->cur_ele_num;
    p_col_buf = p_ctr->col_buf;
    ree_set(p_col_buf, 0, ele_num*ELE_FP_SIZE);

    for (uint32_t col_h_ind = 0; col_h_ind<p_ctr->cur_k_l; col_h_ind++)
    {
        sig_ind_w_padding = col_h_ind;
        for (uint32_t col_w_ind = 0; col_w_ind<p_ctr->cur_out_pack_l; col_w_ind++)
        {
            if (col_w_ind < p_ctr->cur_out_l)
            {
                feature = sig2col_get_deconv_pixel_fp(sig_ind_w_padding, p_mat, p_para);
            } else
            {
                feature = 0.0f;
            }
            *p_col_buf = feature;
            sig_ind_w_padding += p_mat->stride;
            p_col_buf++;
        }
    }
EXIT_SIG2COL_MAT_DECODER_MAT_FP:
    SIG2COL_FUNC_EXIT;
    return retval;
}

void sig2col_ctr_destructor(sig2col_ctr_t *p_ctr)
{
    SIG2COL_FUNC_ENTRANCE;
    ree_check_null_exit(p_ctr, EXIT_SIG2COL_CTR_DESTRUCTOR, "%s occurs error due to p_ctr is NULL", __func__);
    ree_check_true_exit((!p_ctr->inited), EXIT_SIG2COL_CTR_DESTRUCTOR, "%s directly return due to p_ctr->inited is FALSE", __func__);
    ree_free(p_ctr->col_buf);
EXIT_SIG2COL_CTR_DESTRUCTOR:
    SIG2COL_FUNC_EXIT;
}
