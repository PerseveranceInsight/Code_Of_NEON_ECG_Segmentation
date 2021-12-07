#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arm_util.h"
#include "ecg_seg_graph.h"
#include "ecg_seg_signal.h"
#include "ecg_response_def.h"
#include "ecg_seg_util.h"

static ecg_seg_graph_t g_graph_ctr = {0};

int main(int argc, char* argv[])
{

    int32_t retval = ECG_SEG_OK;
    mat_sig_para_t input_sig_para = {.ori_l = ECG_SIGNAL_ORI_L,
                                     .k_l = ECG_SIGNAL_K_L,
                                     .padding = ECG_SIGNAL_PADDING,
                                     .stride = ECG_SIGNAL_STRIDE,};
    ree_log(LOG_DEBUG, "%s starts", __func__);
    retval = ecg_seg_graph_constructor_fp(ECG_SIGNAL_ORI_C, 
                                          ECG_MIDDLE_FEATURE_GROUP_NUM, 
                                          ECG_OUTPUT_PRED_GROUP_NUM, 
                                          ECG_CONV_RELU_FUSE_GROUP_NUM, 
                                          ECG_TRANCONV_GROUP_NUM, &g_graph_ctr);
    retval = ecg_seg_graph_input_constructor_fopen(ECG_SIGNAL,
                                                   &input_sig_para,
                                                   &g_graph_ctr);
    retval = ecg_seg_graph_context_init(&g_graph_ctr);
    retval = ecg_seg_graph_forward(&g_graph_ctr);
    ecg_seg_graph_destructor_fp(&g_graph_ctr);
    ree_log(LOG_DEBUG, "%s ends retval %d", __func__, retval);
    return 0;
}
