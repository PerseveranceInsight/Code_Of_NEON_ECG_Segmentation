#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arm_util.h"
#include "ecg_seg_def.h"
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
    char **signal_path = NULL;
    conv_fuse_relu_t *p_module = NULL;
    signal_container_t *p_signal = NULL;
    sig2col_ctr_t *p_sig2col_ctr = NULL;

    signal_path = ree_malloc(sizeof(char*));
    signal_path[0] = ree_malloc(sizeof(char)*ECG_SEG_PATH_MAX);
    sprintf(signal_path[0], "%s", ECG_TINY_SIGNAL);
    ree_log(LOG_DEBUG, "%s path %s", __func__, signal_path[0]);

    mat_sig_para_t weight_para = {.ori_l = ECG_SEG_ENCODER_CONVRELU_1_K_L,
                                  .k_l = ECG_SEG_ENCODER_CONVRELU_1_K_L,
                                  .padding = ECG_SEG_ENCODER_CONVRELU_1_K_DUMMING_PADDING,
                                  .stride = ECG_SEG_ENCODER_CONVRELU_1_K_DUMMING_STRIDE,};
    mat_sig_para_t signal_para = {.ori_l = ECG_SIGNAL_ORI_L,
                                  .k_l = ECG_SIGNAL_K_L,
                                  .padding = ECG_SIGNAL_PADDING,
                                  .stride = ECG_SIGNAL_STRIDE,};

    conv_fuse_relu_constructor_fopen(&weight_para, &p_module, 
                                     UNET_ENCODER1_CONV1D_BLOCK_1_CONV_WEIGHT0,
                                     UNET_ENCODER1_CONV1D_BLOCK_1_CONV_WEIGHT1,
                                     UNET_ENCODER1_CONV1D_BLOCK_1_CONV_WEIGHT2,
                                     UNET_ENCODER1_CONV1D_BLOCK_1_CONV_WEIGHT3,
                                     UNET_ENCODER1_CONV1D_BLOCK_1_CONV_BIAS);
    retval = signal_container_constructor_fp_fopen(ECG_SIGNAL_ORI_C, 
                                                   &signal_para,
                                                   &p_signal,
                                                   signal_path);
    retval = sig2col_ctr_fp_constructor(ECG_SIG2COL_MAX_OUT_L, 
                                        ECG_SIG2COL_MAX_K_L, 
                                        &p_sig2col_ctr);
    retval = sig2col_mat_fp(p_sig2col_ctr, p_signal->signal);
    sig2col_printf_mat_fp(p_sig2col_ctr);

    ree_log(LOG_DEBUG, "%s ends", __func__);

    conv_fuse_relu_destructor(p_module);
    signal_container_destructor(p_signal);
    sig2col_ctr_destructor(p_sig2col_ctr);
    ree_free(p_module);
    ree_free(p_signal);
    ree_free(p_sig2col_ctr);
    ree_free(signal_path[0]);
    ree_free(signal_path);
    return 0;
}
