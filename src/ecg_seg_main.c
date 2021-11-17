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
#include "ecg_seg_weight.h"
#include "ecg_response_def.h"
#include "ecg_seg_util.h"

int main(int argc, char* argv[])
{
    ree_log(LOG_DEBUG, "%s starts", __func__);

    int32_t retval = ECG_SEG_OK;
    conv_fuse_relu_t *p_module = NULL;
    signal_container_t *p_input_sig = NULL;
    signal_container_t *p_mid_feature = NULL;
    sig2col_ctr_t *p_sig2col_ctr = NULL;

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
    ree_log(LOG_DEBUG, "%s ends", __func__);
    ree_free(p_module);
    ree_free(p_input_sig);
    ree_free(p_mid_feature);
    ree_free(p_sig2col_ctr);
    return 0;
}
