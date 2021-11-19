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

int32_t ecg_seg_graph_input_fopen(char *p_sig_path,
                                  ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_sig_path, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_INPUT_FOPEN,
                               "%s occurs error due to p_sig_path is NULL", __func__);
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_INPUT_FOPEN,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_true_exit_retval((p_graph->inited != TRUE), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_INPUT_FOPEN,
                               "%s occurs error due to p_graph->inited is FALSE", __func__);
    ree_log(GRAPH_LOG, "%s path of signal : %s", __func__, p_sig_path);
EXIT_ECG_SEG_GRAPH_INPUT_FOPEN:
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
    ree_free(p_graph->p_in_sigs);
    ree_free(p_graph->p_mid_features);
    ree_free(p_graph->p_out_pred);
    ree_free(p_graph->p_sig2col_ctr);
EXIT_ECG_SEG_GRAPH_DESTRUCTOR:
    GRAPH_FUNC_EXIT;
    return retval;
}
