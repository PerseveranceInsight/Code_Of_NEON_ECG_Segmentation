#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arm_util.h"
#include "ecg_seg_def.h"
#include "ecg_seg_gemm.h"
#include "ecg_seg_matrix.h"
#include "ecg_seg_model.h"
#include "ecg_seg_sig2col.h"
#include "ecg_seg_signal.h"
#include "ecg_response_def.h"
#include "ecg_seg_util.h"

int main(int argc, char* argv[])
{
    ree_log(LOG_DEBUG, "%s starts", __func__);

    int32_t retval = ECG_SEG_OK;
    char **input_sig_path = NULL;
    conv_fuse_relu_t *p_module = NULL;
    signal_container_t *p_input_sig = NULL;
    signal_container_t *p_mid_feature = NULL;
    sig2col_ctr_t *p_sig2col_ctr = NULL;

    input_sig_path = ree_malloc(sizeof(char*));
    input_sig_path[0] = ree_malloc(sizeof(char)*ECG_SEG_PATH_MAX);
    sprintf(input_sig_path[0], "%s", ECG_TINY_SIGNAL);
    ree_log(LOG_DEBUG, "%s path %s", __func__, input_sig_path[0]);

    mat_sig_para_t weight_para = {.ori_l = ECG_SEG_ENCODER_CONVRELU_1_K_L,
                                  .k_l = ECG_SEG_ENCODER_CONVRELU_1_K_L,
                                  .padding = ECG_SEG_ENCODER_CONVRELU_1_K_DUMMING_PADDING,
                                  .stride = ECG_SEG_ENCODER_CONVRELU_1_K_DUMMING_STRIDE,};
    mat_sig_para_t input_sig_para = {.ori_l = ECG_SIGNAL_ORI_L,
                                     .k_l = ECG_SIGNAL_K_L,
                                     .padding = ECG_SIGNAL_PADDING,
                                     .stride = ECG_SIGNAL_STRIDE,};
    mat_sig_para_t mid_feat_para = {.ori_l = ECG_SIGNAL_MID1_ORI_L,
                                    .k_l = ECG_SIGNAL_MID1_K_L,
                                    .padding = ECG_SIGNAL_MID1_PADDING,
                                    .stride = ECG_SIGNAL_MID1_STRIDE,};

    conv_fuse_relu_constructor_fopen(&weight_para, &p_module, 
                                     UNET_ENCODER1_CONV1D_BLOCK_1_CONV_WEIGHT0,
                                     UNET_ENCODER1_CONV1D_BLOCK_1_CONV_WEIGHT1,
                                     UNET_ENCODER1_CONV1D_BLOCK_1_CONV_WEIGHT2,
                                     UNET_ENCODER1_CONV1D_BLOCK_1_CONV_WEIGHT3,
                                     UNET_ENCODER1_CONV1D_BLOCK_1_CONV_BIAS);
    retval = signal_container_constructor_fp_fopen(ECG_SIGNAL_ORI_C, 
                                                   &input_sig_para,
                                                   &p_input_sig,
                                                   input_sig_path);
    ree_log(LOG_DEBUG, "%s retval of signal_container_constructor_fp_fopen %d", __func__, retval);
    retval = signal_container_constructor(ECG_SIGNAL_MID1_MAX_C,
                                          &mid_feat_para,
                                          &p_mid_feature);
    ree_log(LOG_DEBUG, "%s retval of signal_container_constructor %d", __func__, retval);
    retval = sig2col_ctr_fp_constructor(ECG_SIG2COL_MAX_OUT_L, 
                                        ECG_SIG2COL_MAX_K_L, 
                                        &p_sig2col_ctr);
    ree_log(LOG_DEBUG, "%s retval of sig2col_ctr_fp_constructor %d", __func__, retval);
    retval = sig2col_mat_fp(p_sig2col_ctr, p_input_sig->signal);
    ree_log(LOG_DEBUG, "%s retval of sig2col_mat_fp %d", __func__, retval);

    ecg_seg_fp_gemm(&(p_module->conv_weight[0]),
                    p_sig2col_ctr,
                    &(p_mid_feature->signal[0]));

    ree_log(LOG_DEBUG, "%s ends", __func__);

    conv_fuse_relu_destructor(p_module);
    signal_container_destructor(p_input_sig);
    signal_container_destructor(p_mid_feature);
    sig2col_ctr_destructor(p_sig2col_ctr);
    ree_free(p_module);
    ree_free(p_input_sig);
    ree_free(p_mid_feature);
    ree_free(p_sig2col_ctr);
    ree_free(input_sig_path[0]);
    ree_free(input_sig_path);
    return 0;
}
