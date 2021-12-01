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

static mat_sig_para_t weight_para0_0 = {.ori_l = ECG_SEG_ENCODER_CONVRELU_0_K_L,
                                        .k_l = ECG_SEG_ENCODER_CONVRELU_0_K_L,
                                        .padding = ECG_SEG_ENCODER_CONVRELU_0_K_DUMMING_PADDING,
                                        .stride = ECG_SEG_ENCODER_CONVRELU_0_K_DUMMING_STRIDE,};

static mat_sig_para_t weight_para0_1 = {.ori_l = ECG_SEG_ENCODER_CONVRELU_0_K_L,
                                        .k_l = ECG_SEG_ENCODER_CONVRELU_0_K_L,
                                        .padding = ECG_SEG_ENCODER_CONVRELU_0_K_DUMMING_PADDING,
                                        .stride = ECG_SEG_ENCODER_CONVRELU_0_K_DUMMING_STRIDE,};

static mat_sig_para_t mid_feat_para0 = {.ori_l = ECG_SIGNAL_MID0_ORI_L,
                                        .k_l = ECG_SIGNAL_MID0_K_L,
                                        .padding = ECG_SIGNAL_MID0_PADDING,
                                        .stride = ECG_SIGNAL_MID0_STRIDE,};

static mat_sig_para_t mid_feat_para1 = {.ori_l = ECG_SIGNAL_MID1_ORI_L,
                                        .k_l = ECG_SIGNAL_MID1_K_L,
                                        .padding = ECG_SIGNAL_MID1_PADDING,
                                        .stride = ECG_SIGNAL_MID1_STRIDE,};

static mat_sig_para_t weight_para1_0 = {.ori_l = ECG_SEG_ENCODER_CONVRELU_0_K_L,
                                        .k_l = ECG_SEG_ENCODER_CONVRELU_0_K_L,
                                        .padding = ECG_SEG_ENCODER_CONVRELU_0_K_DUMMING_PADDING,
                                        .stride = ECG_SEG_ENCODER_CONVRELU_0_K_DUMMING_STRIDE,};

static mat_sig_para_t weight_para1_1 = {.ori_l = ECG_SEG_ENCODER_CONVRELU_0_K_L,
                                        .k_l = ECG_SEG_ENCODER_CONVRELU_0_K_L,
                                        .padding = ECG_SEG_ENCODER_CONVRELU_0_K_DUMMING_PADDING,
                                        .stride = ECG_SEG_ENCODER_CONVRELU_0_K_DUMMING_STRIDE,};

static mat_sig_para_t mid_feat_para2 = {.ori_l = ECG_SIGNAL_MID2_ORI_L,
                                        .k_l = ECG_SIGNAL_MID2_K_L,
                                        .padding = ECG_SIGNAL_MID2_PADDING,
                                        .stride = ECG_SIGNAL_MID2_STRIDE,};

static max_pool_parameters_t max_pool_parameters = {.kernel_size = ECG_SIGNAL_MAX_POOL_KERNEL_SIZE,
                                                    .stride = ECG_SIGNAL_MAX_POOL_STRIDE,
                                                    .padding = ECG_SIGNAL_MAX_POOL_PADDING,};

static void ecg_seg_graph_constructor_param(ecg_seg_graph_t *p_graph);
static int32_t ecg_seg_graph_mid_feature0_constructor(mat_sig_para_t *p_sig_para, ecg_seg_graph_t *p_graph);
static int32_t ecg_seg_graph_mid_feature1_constructor(mat_sig_para_t *p_sig_para, ecg_seg_graph_t *p_graph);
static int32_t ecg_seg_graph_mid_feature2_constructor(mat_sig_para_t *p_sig_para, ecg_seg_graph_t *p_graph);
static int32_t ecg_seg_graph_output_constructor(mat_sig_para_t *p_sig_para, ecg_seg_graph_t *p_graph);
static int32_t ecg_seg_graph_sig2col0_constructor(uint32_t max_out_l, uint32_t max_k_l, ecg_seg_graph_t *p_graph);
static int32_t ecg_seg_graph_conv_fuse_relu0_0_constructor(mat_sig_para_t *p_sig_para, ecg_seg_graph_t *p_graph);
static int32_t ecg_seg_graph_conv_fuse_relu0_1_constructor(mat_sig_para_t *p_sig_para, ecg_seg_graph_t *p_graph);
static int32_t ecg_seg_graph_conv_fuse_relu1_0_constructor(mat_sig_para_t *p_sig_para, ecg_seg_graph_t *p_graph);
static int32_t ecg_seg_graph_conv_fuse_relu1_1_constructor(mat_sig_para_t *p_sig_para, ecg_seg_graph_t *p_graph);
static int32_t ecg_seg_graph0_0_forward(ecg_seg_graph_t *p_graph);
static int32_t ecg_seg_graph0_1_forward(ecg_seg_graph_t *p_graph);
static int32_t ecg_seg_graph_max_pool_0_forward(ecg_seg_graph_t *p_graph);
static int32_t ecg_seg_graph1_0_forward(ecg_seg_graph_t *p_graph);
static int32_t ecg_seg_graph1_1_forward(ecg_seg_graph_t *p_graph);
static int32_t ecg_seg_graph_max_pool_1_forward(ecg_seg_graph_t *p_graph);

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
    p_graph->p_modules = ree_malloc(sizeof(conv_fuse_relu_t)*conv_fuse_relu_num);
    ree_check_null_exit_retval(p_graph->p_modules, retval, ECG_SEG_ALLOC_FAILED, EXIT_ECG_SEG_GRAPH_CONSTRUCTOR,
                               "%s occurs error due to alloc p_graph->p_modules failed", __func__);
    ree_set(p_graph->p_modules, 0, sizeof(conv_fuse_relu_t)*conv_fuse_relu_num);
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
    retval = signal_container_constructor_fp(ECG_SIGNAL_MID0_MAX_C,
                                             p_sig_para,
                                             &p_graph->p_mid_features);
EXIT_ECG_SEG_GRAPH_MID_FEATURE0_CONSTRUCTOR:
    GRAPH_FUNC_EXIT;
    return retval;
}

static int32_t ecg_seg_graph_mid_feature1_constructor(mat_sig_para_t *p_sig_para,
                                                      ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    signal_container_t *p_sig_container = NULL;
    ree_check_null_exit_retval(p_sig_para, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_MID_FEATURE1_CONSTRUCTOR,
                               "%s occurs error due to p_sig_para is NULL", __func__);
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_MID_FEATURE1_CONSTRUCTOR,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_true_exit_retval((!p_graph->inited), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MID_FEATURE1_CONSTRUCTOR,
                               "%s occurs error due to p_graph->inited is FALSE", __func__);
    ree_check_true_exit_retval((p_graph->mid_num < 2), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MID_FEATURE1_CONSTRUCTOR,
                               "%s occurs error due to p_graph->mid_num is less than 2", __func__);
    ree_check_null_exit_retval((p_graph->p_mid_features + sizeof(signal_container_t)), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MID_FEATURE1_CONSTRUCTOR,
                               "%s occurs error due to p_graph->p_mid_features + sizeof(signal_container_t) is NULL", __func__);
    p_sig_container = &(p_graph->p_mid_features[1]);
    retval = signal_container_constructor_fp(ECG_SIGNAL_MID1_MAX_C,
                                             p_sig_para,
                                             &p_sig_container);
EXIT_ECG_SEG_GRAPH_MID_FEATURE1_CONSTRUCTOR:
    GRAPH_FUNC_EXIT;
    return retval;
}

static int32_t ecg_seg_graph_mid_feature2_constructor(mat_sig_para_t *p_sig_para,
                                                      ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    signal_container_t *p_sig_container = NULL;
    ree_check_null_exit_retval(p_sig_para, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_MID_FEATURE2_CONSTRUCTOR,
                               "%s occurs error due to p_sig_para is NULL", __func__);
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_MID_FEATURE2_CONSTRUCTOR,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_true_exit_retval((!p_graph->inited), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MID_FEATURE2_CONSTRUCTOR,
                               "%s occurs error due to p_graph->inited is FALSE", __func__);
    ree_check_true_exit_retval((p_graph->mid_num < 3), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MID_FEATURE2_CONSTRUCTOR,
                               "%s occurs errod due to p_graph->mid_num is less than 3", __func__);
    ree_check_null_exit_retval((p_graph->p_mid_features + sizeof(signal_container_t)*2), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MID_FEATURE2_CONSTRUCTOR,
                               "%s occurs error due to p_graph->p_mid_features + sizeof(signal_container_t)*2 is NULL", __func__);
    p_sig_container = &(p_graph->p_mid_features[2]);
    retval = signal_container_constructor_fp(ECG_SIGNAL_MID2_MAX_C,
                                             p_sig_para,
                                             &p_sig_container);
EXIT_ECG_SEG_GRAPH_MID_FEATURE2_CONSTRUCTOR:
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

static int32_t ecg_seg_graph_conv_fuse_relu0_0_constructor(mat_sig_para_t *p_sig_para,
                                                           ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    void **pp_weight_buf = NULL;
    void **pp_bias_buf = NULL;
    ree_check_null_exit_retval(p_sig_para, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU0_0_CONSTRUCTOR,
                               "%s occurs error due to p_sig_para is NULL", __func__);
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU0_0_CONSTRUCTOR,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_true_exit_retval((!p_graph->inited), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU0_0_CONSTRUCTOR,
                               "%s occurs error due to p_graph->inited is FALSE", __func__);
    ree_check_true_exit_retval((p_graph->conv_fuse_relu_num < 1), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU0_0_CONSTRUCTOR,
                               "%s occurs error due to p_graph->conv_fuse_relu_num is less than 1", __func__);
    print_mat_sig_para(p_sig_para);
    pp_weight_buf = ree_malloc(sizeof(void*)*ECG_SEG_ENCODER_CONVRELU0_0_K_C);
    ree_check_null_exit_retval(pp_weight_buf, retval, ECG_SEG_ALLOC_FAILED, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU0_0_CONSTRUCTOR,
                               "%s occurs error due to allocate pp_weight_buf failed", __func__);
    ree_set(pp_weight_buf, 0, sizeof(void*)*ECG_SEG_ENCODER_CONVRELU0_0_K_C);
    pp_weight_buf[0] = (void*)(&unet_encoder0_conv1d_block_0_conv_weight0);
    pp_weight_buf[1] = (void*)(&unet_encoder0_conv1d_block_0_conv_weight1);
    pp_weight_buf[2] = (void*)(&unet_encoder0_conv1d_block_0_conv_weight2);
    pp_weight_buf[3] = (void*)(&unet_encoder0_conv1d_block_0_conv_weight3);
    pp_bias_buf = (void*)&unet_encoder0_conv1d_block_0_conv_bias;
    ree_log(GRAPH_LOG, "%s %p %p %p %p", __func__, pp_weight_buf[0], pp_weight_buf[1], pp_weight_buf[2], pp_weight_buf[3]);
    ree_log(GRAPH_LOG, "%s pp_bias_buf %p unet_encoder0_conv1d_block_0_conv_bias %p", __func__, pp_bias_buf, &unet_encoder0_conv1d_block_0_conv_bias);
    retval = conv_fuse_relu_constructor_static(ECG_SEG_ENCODER_CONVRELU0_0_K_C,
                                               p_sig_para,
                                               &(p_graph->p_modules),
                                               pp_weight_buf,
                                               pp_bias_buf);
    ree_log(GRAPH_LOG, "%s retval of conv_fuse_relu_constructor_static %d", __func__, retval); 
EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU0_0_CONSTRUCTOR:
    ree_free(pp_weight_buf);
    GRAPH_FUNC_EXIT;
    return retval;
}

static int32_t ecg_seg_graph_conv_fuse_relu0_1_constructor(mat_sig_para_t *p_sig_para,
                                                           ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    void **pp_weight_buf = NULL;
    void **pp_bias_buf = NULL;
    conv_fuse_relu_t *p_module = NULL;
    ree_check_null_exit_retval(p_sig_para, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU0_1_CONSTRUCTOR,
                               "%s occurs error due to p_sig_para is NULL", __func__);
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU0_1_CONSTRUCTOR,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_true_exit_retval((!p_graph->inited), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU0_1_CONSTRUCTOR,
                               "%s occurs error due to p_graph->inited is FALSE", __func__);
    ree_check_true_exit_retval((p_graph->conv_fuse_relu_num < 2), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU0_1_CONSTRUCTOR,
                               "%s occurs error due to p_graph->conv_fuse_relu_num is less than 2", __func__);
    print_mat_sig_para(p_sig_para);
    pp_weight_buf = ree_malloc(sizeof(void*)*ECG_SEG_ENCODER_CONVRELU0_1_K_C);
    ree_check_null_exit_retval(pp_weight_buf, retval, ECG_SEG_ALLOC_FAILED, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU0_1_CONSTRUCTOR,
                               "%s occurs error due to allocate pp_weight_buf failed", __func__);
    ree_set(pp_weight_buf, 0, sizeof(void*)*ECG_SEG_ENCODER_CONVRELU0_1_K_C);
    pp_weight_buf[0] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight0_0);
    pp_weight_buf[1] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight0_1);
    pp_weight_buf[2] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight0_2);
    pp_weight_buf[3] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight0_3);
    ree_log(GRAPH_LOG, "%s pp_weight_buf %p %p %p %p", __func__, pp_weight_buf[0], pp_weight_buf[1], pp_weight_buf[2], pp_weight_buf[3]); 
    pp_weight_buf[4] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight1_0);
    pp_weight_buf[5] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight1_1);
    pp_weight_buf[6] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight1_2);
    pp_weight_buf[7] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight1_3);
    ree_log(GRAPH_LOG, "%s pp_weight_buf %p %p %p %p", __func__, pp_weight_buf[3], pp_weight_buf[4], pp_weight_buf[5], pp_weight_buf[6]);
    pp_weight_buf[8] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight2_0);
    pp_weight_buf[9] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight2_1);
    pp_weight_buf[10] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight2_2);
    pp_weight_buf[11] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight2_3);
    ree_log(GRAPH_LOG, "%s pp_weight_buf %p %p %p %p", __func__, pp_weight_buf[7], pp_weight_buf[8], pp_weight_buf[9], pp_weight_buf[10]);
    pp_weight_buf[12] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight3_0);
    pp_weight_buf[13] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight3_1);
    pp_weight_buf[14] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight3_2);
    pp_weight_buf[15] = (void*)(&unet_encoder0_conv1d_block_1_conv_weight3_3);
    ree_log(GRAPH_LOG, "%s pp_weight_buf %p %p %p %p", __func__, pp_weight_buf[12], pp_weight_buf[13], pp_weight_buf[14], pp_weight_buf[15]);
    pp_bias_buf = (void*)(&unet_encoder0_conv1d_block_1_conv_bias);
    ree_log(GRAPH_LOG, "%s pp_bias_buf %p unet_encoder0_conv1d_block_1_conv_bias %p", __func__, pp_bias_buf, &unet_encoder0_conv1d_block_1_conv_bias);
    p_module = &p_graph->p_modules[1];
    retval = conv_fuse_relu_constructor_static(ECG_SEG_ENCODER_CONVRELU0_1_K_C,
                                               p_sig_para,
                                               &(p_module),
                                               pp_weight_buf,
                                               pp_bias_buf);

EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU0_1_CONSTRUCTOR:
    ree_free(pp_weight_buf);
    GRAPH_FUNC_EXIT;
    return retval;
}

static int32_t ecg_seg_graph_conv_fuse_relu1_0_constructor(mat_sig_para_t *p_sig_para,
                                                           ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    void **pp_weight_buf = NULL;
    void **pp_bias_buf = NULL;
    conv_fuse_relu_t *p_module = NULL;
    float *p_weight = NULL;
    ree_check_null_exit_retval(p_sig_para, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU1_0_CONSTRUCTOR,
                               "%s occurs error due to p_sig_para is NULL", __func__);
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU1_0_CONSTRUCTOR,
                               "%s occurs error due to p_sig_para is NULL", __func__);
    ree_check_true_exit_retval((!p_graph->inited), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU1_0_CONSTRUCTOR,
                               "%s occurs error due to p_graph->inited is FALSE", __func__);
    ree_check_true_exit_retval((p_graph->conv_fuse_relu_num < 3), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU1_0_CONSTRUCTOR,
                               "%s occurs error due to p_graph->conv_fuse_relu_num is less than 3", __func__);
    print_mat_sig_para(p_sig_para);
    pp_weight_buf = ree_malloc(sizeof(void*)*ECG_SEG_ENCODER_CONVRELU1_0_K_C);
    ree_check_null_exit_retval(pp_weight_buf, retval, ECG_SEG_ALLOC_FAILED, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU1_0_CONSTRUCTOR,
                               "%s occurs error due to allocate pp_weight_buf failed", __func__);
    ree_set(pp_weight_buf, 0, sizeof(void*)*ECG_SEG_ENCODER_CONVRELU1_0_K_C);
    p_weight = (float*)unet_encoder1_conv1d_block_0_conv1d_weight;
    for (uint32_t ch_ind = 0; ch_ind < ECG_SEG_ENCODER_CONVRELU1_0_K_C; ch_ind++)
    {
        pp_weight_buf[ch_ind] = (void*)(p_weight);
        ree_log(GRAPH_LOG, "%s ch_ind %d %p", __func__, ch_ind, pp_weight_buf[ch_ind]);
        p_weight += ECG_SEG_ENCODER_CONVRELU_WEIGHT_PACK_SIZE;
    }
    pp_bias_buf = (void*)(&unet_encoder1_conv1d_block_0_conv1d_bias);
    ree_log(GRAPH_LOG, "%s pp_bias_buf %p unet_encoder1_conv1d_block_1_conv_bias %p", __func__, pp_bias_buf, &unet_encoder1_conv1d_block_0_conv1d_bias);
    p_module = &p_graph->p_modules[2];
    retval = conv_fuse_relu_constructor_static(ECG_SEG_ENCODER_CONVRELU1_0_K_C,
                                               p_sig_para,
                                               &(p_module), 
                                               pp_weight_buf,
                                               pp_bias_buf);
EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU1_0_CONSTRUCTOR:
    ree_free(pp_weight_buf);
    GRAPH_FUNC_EXIT;
    return retval;
}

static int32_t ecg_seg_graph_conv_fuse_relu1_1_constructor(mat_sig_para_t *p_sig_para,
                                                           ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    void **pp_weight_buf = NULL;
    void **pp_bias_buf = NULL;
    conv_fuse_relu_t *p_module = NULL;
    float *p_weight = NULL;
    ree_check_null_exit_retval(p_sig_para, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU1_1_CONSTRUCTOR,
                               "%s occurs error due to p_sig_para is NULL", __func__);
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU1_1_CONSTRUCTOR,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_true_exit_retval((!p_graph->inited), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU1_1_CONSTRUCTOR,
                               "%s occurs error due to p_graph->inited is FALSE", __func__);
    ree_check_true_exit_retval((p_graph->conv_fuse_relu_num < 4), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU1_1_CONSTRUCTOR,
                               "%s occurs error due to p_graph->conv_fuse_relu_num is less than 4", __func__);
    print_mat_sig_para(p_sig_para);
    pp_weight_buf = ree_malloc(sizeof(void*)*ECG_SEG_ENCODER_CONVRELU1_1_K_C);
    ree_check_null_exit_retval(pp_weight_buf, retval, ECG_SEG_ALLOC_FAILED, EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU1_1_CONSTRUCTOR,
                               "%s occurs error due to allocate pp_weight_buf failed", __func__);
    ree_set(pp_weight_buf, 0, sizeof(void*)*ECG_SEG_ENCODER_CONVRELU1_1_K_C);
    p_weight = (float*)unet_encoder1_conv1d_block_1_conv1d_weight;
    for (uint32_t ch_ind = 0; ch_ind < ECG_SEG_ENCODER_CONVRELU1_1_K_C; ch_ind++)
    {
        pp_weight_buf[ch_ind] = (void*)(p_weight);
        ree_log(GRAPH_LOG, "%s ch_ind %d %p", __func__, ch_ind, pp_weight_buf[ch_ind]);
        p_weight += ECG_SEG_ENCODER_CONVRELU_WEIGHT_PACK_SIZE;
    }
    pp_bias_buf = (void*)(&unet_encoder1_conv1d_block_1_conv1d_bias);
    ree_log(GRAPH_LOG, "%s pp_bias_buf %p unet_encoder1_cond1d_block_1_conv1d_bias %p", __func__, 
                                                                                        pp_bias_buf,
                                                                                        &unet_encoder1_conv1d_block_1_conv1d_bias);
    p_module = &p_graph->p_modules[3];
    retval = conv_fuse_relu_constructor_static(ECG_SEG_ENCODER_CONVRELU1_1_K_C,
                                               p_sig_para,
                                               &(p_module),
                                               pp_weight_buf,
                                               pp_bias_buf);
EXIT_ECG_SEG_GRAPH_CONV_FUSE_RELU1_1_CONSTRUCTOR:
    ree_free(pp_weight_buf);
    GRAPH_FUNC_EXIT;
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
    ree_check_true_exit_retval((retval != ECG_SEG_OK), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONTEXT_INIT,
                               "%s occurs error due to retval != ECG_SEG_OK", __func__);
    retval = ecg_seg_graph_sig2col0_constructor(ECG_SIG2COL_MAX_OUT_L,
                                                ECG_SIG2COL_MAX_K_L,
                                                p_graph);
    ree_check_true_exit_retval((retval != ECG_SEG_OK), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONTEXT_INIT,
                               "%s occurs error due to retval != ECG_SEG_OK", __func__);
    retval = ecg_seg_graph_conv_fuse_relu0_0_constructor(&weight_para0_0, p_graph);
    ree_check_true_exit_retval((retval != ECG_SEG_OK), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONTEXT_INIT,
                               "%s occurs error due to retval != ECG_SEG_OK", __func__);
    retval = ecg_seg_graph_conv_fuse_relu0_1_constructor(&weight_para0_1, p_graph);
    ree_check_true_exit_retval((retval != ECG_SEG_OK), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONTEXT_INIT,
                               "%s occurs error due to retval != ECG_SEG_OK", __func__);
    retval = ecg_seg_graph_mid_feature1_constructor(&mid_feat_para1, p_graph);
    ree_check_true_exit_retval((retval != ECG_SEG_OK), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONTEXT_INIT,
                               "%s occurs error due to retval != ECG_SEG_OK", __func__);
    retval = ecg_seg_graph_conv_fuse_relu1_0_constructor(&weight_para1_0, p_graph);
    ree_check_true_exit_retval((retval != ECG_SEG_OK), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONTEXT_INIT,
                               "%s occurs error due to retval != ECG_SEG_OK", __func__);
    retval = ecg_seg_graph_conv_fuse_relu1_1_constructor(&weight_para1_1, p_graph);
    ree_check_true_exit_retval((retval != ECG_SEG_OK), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONTEXT_INIT,
                               "%s occurs error due to retval != ECG_SEG_OK", __func__);
    retval = ecg_seg_graph_mid_feature2_constructor(&mid_feat_para2, p_graph);
    ree_check_true_exit_retval((retval != ECG_SEG_OK), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_CONTEXT_INIT,
                               "%s occurs error due to retval != ECG_SEG_OK", __func__);
EXIT_ECG_SEG_GRAPH_CONTEXT_INIT:
    GRAPH_FUNC_EXIT;
    return retval;
}

static int32_t ecg_seg_graph0_0_forward(ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH0_0_FORWARD,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_null_exit_retval(p_graph->p_in_sigs, retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH0_0_FORWARD,
                               "%s occurs error due to p_graph->p_in_sigs is NULL", __func__);
    ree_check_null_exit_retval(p_graph->p_sig2col_ctr, retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH0_0_FORWARD,
                               "%s occurs error due to p_graph->p_sig2col_ctr is NULL", __func__);
    ree_check_null_exit_retval(p_graph->p_mid_features, retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH0_0_FORWARD,
                               "%s occurs error due to p_graph->p_mid_features is NULL", __func__);
    ree_check_null_exit_retval(p_graph->p_modules, retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH0_0_FORWARD,
                               "%s occurs error due to p_graph->p_modules is NULL", __func__);
    retval = conv_fuse_relu_forward(p_graph->p_modules,
                                    p_graph->p_sig2col_ctr,
                                    p_graph->p_in_sigs,
                                    p_graph->p_mid_features,
                                    ECG_SIGNAL_ORI_C,
                                    ECG_SIGNAL_ORI_IN_IND,
                                    ECG_SIGNAL_MID0_0_ORI_C,
                                    ECG_SIGNAL_MID0_0_OUT_IND);
EXIT_ECG_SEG_GRAPH0_0_FORWARD:
    GRAPH_FUNC_EXIT;
    return retval;
}

static int32_t ecg_seg_graph0_1_forward(ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH0_1_FORWARD,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_null_exit_retval(p_graph->p_sig2col_ctr, retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH0_1_FORWARD,
                               "%s occurs error due to p_graph->p_sig2col_ctr is NULL", __func__);
    ree_check_null_exit_retval(p_graph->p_mid_features, retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH0_1_FORWARD,
                               "%s occurs error due to p_graph->p_mid_features is NULL", __func__);
    ree_check_null_exit_retval(&(p_graph->p_modules[1]), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH0_1_FORWARD,
                               "%s occurs error due to &(p_graph->p_modules[1]) is NULL", __func__);
    retval = conv_fuse_relu_forward(&(p_graph->p_modules[1]),
                                    p_graph->p_sig2col_ctr,
                                    p_graph->p_mid_features,
                                    p_graph->p_mid_features,
                                    ECG_SIGNAL_MID0_0_ORI_C,
                                    ECG_SIGNAL_MID0_1_IN_IND,
                                    ECG_SIGNAL_MID0_1_ORI_C,
                                    ECG_SIGNAL_MID0_1_OUT_IND);
EXIT_ECG_SEG_GRAPH0_1_FORWARD:
    GRAPH_FUNC_EXIT;
    return retval;
}

static int32_t ecg_seg_graph_max_pool_0_forward(ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    signal_container_t *p_mid0_feature = NULL;
    signal_container_t *p_mid1_feature = NULL;
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_MAX_POOL_0_FORWARD,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_true_exit_retval((!p_graph->inited), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MAX_POOL_0_FORWARD,
                               "%s occurs error due to p_graph->inited is NULL", __func__);
    ree_check_true_exit_retval((p_graph->mid_num<2), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MAX_POOL_0_FORWARD,
                               "%s occurs error due to p_graph->mid_num %d is less than 2", __func__, p_graph->mid_num);
    ree_check_null_exit_retval((p_graph->p_mid_features), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MAX_POOL_0_FORWARD,
                               "%s occurs error due to p_graph->p_mid_Features is NULL", __func__);
    p_mid0_feature = &(p_graph->p_mid_features[0]);
    p_mid1_feature = &(p_graph->p_mid_features[1]);
    ree_check_null_exit_retval(p_mid0_feature, retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MAX_POOL_0_FORWARD,
                               "%s occurs erorr due to p_graph->p_mid_features[0] is NULL", __func__);
    ree_check_null_exit_retval(p_mid1_feature, retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MAX_POOL_0_FORWARD,
                               "%s occurs erorr due to p_graph->p_mid_features[1] is NULL", __func__);
    retval = max_pool_forward(&max_pool_parameters,
                              p_mid0_feature,
                              p_mid1_feature,
                              ECG_SIGNAL_MID0_1_ORI_C,
                              ECG_SIGNAL_MID0_1_OUT_IND,
                              ECG_SIGNAL_MID1_INPUT_C,
                              ECG_SINGAL_MID1_INPUT_IND);
EXIT_ECG_SEG_GRAPH_MAX_POOL_0_FORWARD:
    GRAPH_FUNC_EXIT;
    return retval;
}

static int32_t ecg_seg_graph1_0_forward(ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    signal_container_t *p_mid1_feature = NULL;
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH1_0_FORWARD,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_null_exit_retval(p_graph->p_sig2col_ctr, retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH1_0_FORWARD,
                               "%s occurs error due to p_graph->p_sig2col_ctr is NULL", __func__);
    ree_check_null_exit_retval(p_graph->p_mid_features, retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH1_0_FORWARD,
                               "%s occurs error due to p_graph->p_mid_features is NULL", __func__);
    ree_check_null_exit_retval(&(p_graph->p_modules[2]), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH1_0_FORWARD,
                               "%s occurs error due to p_graph->p_modules[2] is NUL", __func__);
    p_mid1_feature = &p_graph->p_mid_features[1];
    retval = conv_fuse_relu_forward(&(p_graph->p_modules[2]),
                                    p_graph->p_sig2col_ctr,
                                    p_mid1_feature,
                                    p_mid1_feature,
                                    ECG_SIGNAL_MID1_INPUT_C,
                                    ECG_SINGAL_MID1_INPUT_IND,
                                    ECG_SIGNAL_MID1_0_OUTPUT_C,
                                    ECG_SIGNAL_MID1_0_OUTPUT_IND);
EXIT_ECG_SEG_GRAPH1_0_FORWARD:
    GRAPH_FUNC_EXIT;
    return retval;
}

static int32_t ecg_seg_graph1_1_forward(ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    signal_container_t *p_mid1_feature = NULL;
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH1_1_FORWARD,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_null_exit_retval(p_graph->p_sig2col_ctr, retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH1_1_FORWARD,
                               "%s occurs error due to p_graph->p_sig2col_ctr is NULL", __func__);
    ree_check_null_exit_retval(p_graph->p_mid_features, retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH1_1_FORWARD,
                               "%s occurs error due to p_graph->p_mid_features is NULL", __func__);
    ree_check_null_exit_retval(&(p_graph->p_modules[3]), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH1_1_FORWARD,
                               "%s occurs error due to p_graph->p_modules[3] is NULL", __func__);
    p_mid1_feature = &p_graph->p_mid_features[1];
    retval = conv_fuse_relu_forward(&(p_graph->p_modules[3]),
                                    p_graph->p_sig2col_ctr,
                                    p_mid1_feature,
                                    p_mid1_feature,
                                    ECG_SIGNAL_MID1_0_OUTPUT_C,
                                    ECG_SIGNAL_MID1_0_OUTPUT_IND,
                                    ECG_SIGNAL_MID1_1_OUTPUT_C,
                                    ECG_SIGNAL_MID1_1_OUTPUT_IND);
    GRAPH_FUNC_EXIT;
EXIT_ECG_SEG_GRAPH1_1_FORWARD:
    return retval;
}

static int32_t ecg_seg_graph_max_pool_1_forward(ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    signal_container_t *p_mid1_feature = NULL;
    signal_container_t *p_mid2_feature = NULL;
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_MAX_POOL_1_FORWARD,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_true_exit_retval((!p_graph->inited), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MAX_POOL_1_FORWARD,
                               "%s occurs error due to p_graph->inited is NULL", __func__);
    ree_check_true_exit_retval((p_graph->mid_num < 3), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MAX_POOL_1_FORWARD,
                               "%s occurs error due to p_graph->mid_num %d is less than 3", __func__, p_graph->mid_num);
    ree_check_null_exit_retval((p_graph->p_mid_features), retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MAX_POOL_1_FORWARD,
                               "%s occurs error due to p_graph->p_mid_features is NULL", __func__);
    p_mid1_feature = &(p_graph->p_mid_features[1]);
    p_mid2_feature = &(p_graph->p_mid_features[2]);
    ree_check_null_exit_retval(p_mid1_feature, retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MAX_POOL_1_FORWARD,
                               "%s occurs error due to p_graph->p_mid1_features[1] is NULL", __func__);
    ree_check_null_exit_retval(p_mid2_feature, retval, ECG_SEG_ERROR_STATE, EXIT_ECG_SEG_GRAPH_MAX_POOL_1_FORWARD,
                               "%s occurs error due to p_graph->p_mid1_features[2] is NULL", __func__);
    retval = max_pool_forward(&max_pool_parameters,
                              p_mid1_feature,
                              p_mid2_feature,
                              ECG_SIGNAL_MID1_1_OUTPUT_C,
                              ECG_SIGNAL_MID1_1_OUTPUT_IND,
                              ECG_SIGNAL_MID2_INPUT_C,
                              ECG_SIGNAL_MID2_INPUT_IND);
EXIT_ECG_SEG_GRAPH_MAX_POOL_1_FORWARD:
    GRAPH_FUNC_EXIT;
    return retval;
}

int32_t ecg_seg_graph_forward(ecg_seg_graph_t *p_graph)
{
    GRAPH_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_graph, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_FORWARD,
                               "%s occurs error due to p_graph is NULL", __func__);
    ree_check_true_exit_retval((!p_graph->inited), retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_GRAPH_FORWARD,
                               "%s occurs error due to p_graph->inited is FALSE", __func__);
    retval = ecg_seg_graph0_0_forward(p_graph);
    ree_log(GRAPH_LOG, "%s retval of ecg_seg_graph0_0_forward %d", __func__, retval);
    ree_check_true_exit((retval != ECG_SEG_OK), EXIT_ECG_SEG_GRAPH_FORWARD, "%s occurs error due to retval != ECG_SEG_OK", __func__);
    retval = ecg_seg_graph0_1_forward(p_graph);
    ree_log(GRAPH_LOG, "%s retval of ecg_seg_graph0_1_forward %d", __func__, retval);
    ree_check_true_exit((retval != ECG_SEG_OK), EXIT_ECG_SEG_GRAPH_FORWARD, "%s occurs error due to retval != ECG_SEG_OK", __func__);
    retval = ecg_seg_graph_max_pool_0_forward(p_graph);
    ree_log(GRAPH_LOG, "%s retval of ecg_seg_graph_max_pool_0_forward %d", __func__, retval);
    ree_check_true_exit((retval != ECG_SEG_OK), EXIT_ECG_SEG_GRAPH_FORWARD, "%s occurs error due to retval != ECG_SEG_OK", __func__);
    retval = ecg_seg_graph1_0_forward(p_graph);
    ree_log(GRAPH_LOG, "%s retval of ecg_seg_graph1_0_forward %d", __func__, retval);
    ree_check_true_exit((retval != ECG_SEG_OK), EXIT_ECG_SEG_GRAPH_FORWARD, "%s occurs error due to retval != ECG_SEG_OK", __func__);
    retval = ecg_seg_graph1_1_forward(p_graph);
    ree_log(GRAPH_LOG, "%s retval of ecg_seg_graph1_1_forward %d", __func__, retval);
    ree_check_true_exit((retval != ECG_SEG_OK), EXIT_ECG_SEG_GRAPH_FORWARD, "%s occurs error due to retval != ECG_SEG_OK", __func__);
    retval = ecg_seg_graph_max_pool_1_forward(p_graph);
    ree_log(GRAPH_LOG, "%s retval of ecg_seg_graph_max_pool_1_forward %d", __func__, retval);
    ree_check_true_exit((retval != ECG_SEG_OK), EXIT_ECG_SEG_GRAPH_FORWARD, "%s occurs error due to retval != ECG_SEG_OK", __func__);
EXIT_ECG_SEG_GRAPH_FORWARD:
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
    for (uint32_t ind = 0; ind < p_graph->conv_fuse_relu_num; ind++)
    {
        conv_fuse_relu_destructor(&(p_graph->p_modules[ind]));
    }
    signal_container_destructor(p_graph->p_out_pred);
    ree_free(p_graph->p_in_sigs);
    ree_free(p_graph->p_mid_features);
    ree_free(p_graph->p_out_pred);
    ree_free(p_graph->p_sig2col_ctr);
    ree_free(p_graph->p_modules);
EXIT_ECG_SEG_GRAPH_DESTRUCTOR:
    GRAPH_FUNC_EXIT;
    return retval;
}
