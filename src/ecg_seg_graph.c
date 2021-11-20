#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "arm_typedef.h"
#include "arm_util.h"

#include "ecg_seg_def.h"
#include "ecg_seg_gemm.h"
#include "ecg_seg_graph.h"
#include "ecg_seg_matrix.h"
#include "ecg_seg_model.h"
#include "ecg_seg_sig2col.h"
#include "ecg_seg_signal.h"
#include "ecg_seg_weight.h"

#include "ecg_response_def.h"
#include "ecg_seg_util.h"

static mat_sig_para_t mid_feat_para0 = {.ori_l = ECG_SIGNAL_MID1_ORI_L,
                                        .k_l = ECG_SIGNAL_MID1_K_L,
                                        .padding = ECG_SIGNAL_MID1_PADDING,
                                        .stride = ECG_SIGNAL_MID1_STRIDE,};

static void ecg_seg_graph_constructor_param(ecg_seg_graph_t *p_graph)
{
    ree_log(GRAPH_LOG, "%s in_num %d", __func__, p_graph->in_num);
    ree_log(GRAPH_LOG, "%s mid_num %d", __func__, p_graph->mid_num);
    ree_log(GRAPH_LOG, "%s out_num %d", __func__, p_graph->out_num);
    ree_log(GRAPH_LOG, "%s conv_fuse_relu_num %d", __func__, p_graph->conv_fuse_relu_num);
}

int32_t ecg_seg_graph_constructor_fp(uint32_t in_num,
                                     uint32_t mid_num,
                                     uint32_t out_num,
                                     uint32_t conv_fuse_relu_num,
                                     ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_true_exit_retval((in_num == 0), retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_CONSTRUCTOR,
                               "%s occurs error due to in_num==0", __func__);
    ree_check_true_exit_retval((mid_num == 0), retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_CONSTRUCTOR,
                               "%s occurs error due to mid_num==0", __func__);
    ree_check_true_exit_retval((out_num == 0), retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_CONSTRUCTOR,
                               "%s occurs error due to out_num==0", __func__);
    ree_check_true_exit_retval((conv_fuse_relu_num == 0), retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_CONSTRUCTOR,
                               "%s occurs error due to conv_fuse_relu_num==0", __func__);
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_CONSTRUCTOR,
                               "%s occurs error due to p_graph is NULL", __func__);
    p_graph->in_num = in_num;
    p_graph->mid_num = mid_num;
    p_graph->out_num = out_num;
    p_graph->conv_fuse_relu_num = conv_fuse_relu_num;
    ecg_seg_graph_constructor_param(p_graph);
    p_graph->p_in_sigs = ree_malloc(sizeof(signal_container_t)*in_num);
    ree_check_null_exit_retval(p_graph->p_in_sigs, retval, ECG_SEG_ALLOC_FAILED, EXIT_ECG_SEG_GRAPH_CONSTRUCTOR,
                               "%s occurs error due to alloc p_graph->p_in_sigs failed", __func__);
    ree_set(p_graph->p_in_sigs, 0, sizeof(signal_container_t)*in_num);
    p_graph->p_mid_features = ree_malloc(sizeof(signal_container_t)*mid_num);
    ree_check_null_exit_retval(p_graph->p_mid_features, retval, ECG_SEG_ALLOC_FAILED, EXIT_ECG_SEG_GRAPH_CONSTRUCTOR,
                               "%s occurs error due to alloca p_graph->p_mid_features failed", __func__);
    ree_set(p_graph->p_mid_features, 0, sizeof(signal_container_t)*mid_num);
    p_graph->p_out_pred = ree_malloc(sizeof(signal_container_t)*out_num);
    ree_check_null_exit_retval(p_graph->p_out_pred, retval, ECG_SEG_ALLOC_FAILED, EXIT_ECG_SEG_GRAPH_CONSTRUCTOR,
                               "%s occurs error due to alloc p_graph->p_out_pred failed", __func__);
    ree_set(p_graph->p_out_pred, 0, sizeof(signal_container_t)*out_num);
    p_graph->p_sig2col_ctr = ree_malloc(sizeof(sig2col_ctr_t)*mid_num);
    ree_check_null_exit_retval(p_graph->p_sig2col_ctr, retval, ECG_SEG_ALLOC_FAILED, EXIT_ECG_SEG_GRAPH_CONSTRUCTOR,
                               "%s occurs error due to alloc p_graph->p_sig2col_ctr failed", __func__);
    ree_set(p_graph->p_sig2col_ctr, 0, sizeof(sig2col_ctr_t)*mid_num);
    p_graph->inited = TRUE;
EXIT_ECG_SEG_GRAPH_CONSTRUCTOR:
    if (!p_graph->inited) 
    {
        ree_free(p_graph->p_in_sigs);
        ree_free(p_graph->p_mid_features);
        ree_free(p_graph->p_out_pred);
        ree_free(p_graph->p_sig2col_ctr);
    }
    GRAPH_FUNC_EXIT;
    return retval;
}

int32_t ecg_seg_graph_input_constructor_fopen(char *p_sig_path,
                                              mat_sig_para_t *p_sig_para,
                                              ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_sig_path, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_INPUT_FOPEN,
                               "%s occurs error due to p_sig_path is NULL", __func__);
    ree_check_null_exit_retval(p_sig_para, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_INPUT_FOPEN,
                               "%s occurs error due to p_sig_para is NULL", __func__);
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_INPUT_FOPEN,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_true_exit_retval((p_graph->inited != TRUE), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_INPUT_FOPEN,
                               "%s occurs error due to p_graph->inited is FALSE", __func__);
    ree_log(GRAPH_LOG, "%s path of signal : %s", __func__, p_sig_path);
    ree_check_null_exit_retval((p_graph->p_in_sigs), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_INPUT_FOPEN,
                               "%s occurs error due to p_graph->p_in_sigs is NULL", __func__);
    retval = signal_container_constructor_fp_fopen(p_graph->in_num,
                                                   p_sig_para,
                                                   &p_graph->p_in_sigs,
                                                   &p_sig_path);
EXIT_ECG_SEG_GRAPH_INPUT_FOPEN:
    GRAPH_FUNC_EXIT;
    return retval;
}

static int32_t ecg_seg_graph_mid_feature0_constructor(mat_sig_para_t *p_sig_para,
                                                      ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_sig_para, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_MID_FEATURE0_CONSTRUCTOR,
                               "%s occurs error due to p_sig_para is NULL", __func__);
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_MID_FEATURE0_CONSTRUCTOR,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_true_exit_retval((!p_graph->inited), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MID_FEATURE0_CONSTRUCTOR,
                               "%s occurs error due to p_graph->inited is FALSE", __func__);
    ree_check_null_exit_retval(p_graph->p_mid_features, retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MID_FEATURE0_CONSTRUCTOR,
                               "%s occurs error due to p_graph->p_mid_features is NULL", __func__);
    ree_check_true_exit_retval((p_graph->mid_num < 1), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MID_FEATURE0_CONSTRUCTOR,
                               "%s occurs error due to p_graph->mid_num is less than 1", __func__);
    retval = signal_container_constructor_fp(ECG_SIGNAL_MID1_MAX_C,
                                             p_sig_para,
                                             &p_graph->p_mid_features);
EXIT_ECG_SEG_GRAPH_MID_FEATURE0_CONSTRUCTOR:
    GRAPH_FUNC_EXIT;
    return retval;
}

static int32_t ecg_seg_graph_output_constructor(mat_sig_para_t *p_sig_para,
                                                ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_sig_para, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_OUTPUT,
                               "%s occurs error due to p_sig_para is NULL", __func__);
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_OUTPUT,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_true_exit_retval((p_graph->inited != TRUE), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_OUTPUT,
                               "%s occurs error due to p_graph->inited is FALSE", __func__);
    ree_check_null_exit_retval((p_graph->p_out_pred), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_OUTPUT,
                               "%s occurs error due to p_graph->p_out_pred is NULL", __func__);
    retval = signal_container_constructor_fp(p_graph->out_num,
                                             p_sig_para,
                                             &p_graph->p_out_pred);
EXIT_ECG_SEG_GRAPH_OUTPUT:
    GRAPH_FUNC_EXIT;
    return retval;
}

static int32_t ecg_seg_graph_sig2col0_constructor(uint32_t max_out_l,
                                                  uint32_t max_k_l,
                                                  ecg_seg_graph_t *p_graph)
{
    GEMM_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_true_exit_retval((max_out_l == 0), retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_SIG2COL0_CONSTRUCTOR,
                               "%s directly return due to max_out_l == 0", __func__);
    ree_check_true_exit_retval((max_k_l == 0), retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_SIG2COL0_CONSTRUCTOR,
                               "%s directly return due to max_k_l == 0", __func__);
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_SIG2COL0_CONSTRUCTOR,
                               "%s directly return due to p_graph is NULL", __func__);
    ree_log(GRAPH_LOG, "%s max_out_l %d max_k_l %d", __func__, max_out_l, max_k_l);
    retval = sig2col_ctr_fp_constructor(max_out_l,
                                        max_k_l,
                                        &p_graph->p_sig2col_ctr);
    ree_log(GRAPH_LOG, "%s retval of sig2col_ctr_fp_constructor %d", __func__, retval);
EXIT_ECG_SEG_SIG2COL0_CONSTRUCTOR:
    GEMM_FUNC_EXIT;
    return retval;
}

int32_t ecg_seg_graph_context_init(ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_CONTEXT_INIT,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_true_exit_retval((p_graph->inited != TRUE), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONTEXT_INIT,
                               "%s occurs error due to p_graph->inited is FALSE", __func__);
    retval = ecg_seg_graph_mid_feature0_constructor(&mid_feat_para0, p_graph);
    retval = ecg_seg_graph_sig2col0_constructor(ECG_SIG2COL_MAX_OUT_L,
                                                ECG_SIG2COL_MAX_K_L,
                                                p_graph);
EXIT_ECG_SEG_GRAPH_CONTEXT_INIT:
    GRAPH_FUNC_EXIT;
    return retval;
}

int32_t ecg_seg_graph_destructor_fp(ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_DESTRUCTOR,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_true_exit_retval((p_graph->inited != TRUE), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_DESTRUCTOR,
                               "%s occurs error due to p_graph->inited is FALSE", __func__);
    signal_container_destructor(p_graph->p_in_sigs);
    for (uint32_t ind = 0; ind < p_graph->mid_num; ind++)
    {
        signal_container_destructor(&p_graph->p_mid_features[ind]);
        sig2col_ctr_destructor(&p_graph->p_sig2col_ctr[ind]);
    }
    signal_container_destructor(p_graph->p_out_pred);
    ree_free(p_graph->p_in_sigs);
    ree_free(p_graph->p_mid_features);
    ree_free(p_graph->p_out_pred);
    ree_free(p_graph->p_sig2col_ctr);
EXIT_ECG_SEG_GRAPH_DESTRUCTOR:
    GRAPH_FUNC_EXIT;
    return retval;
}