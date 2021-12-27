#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arm_util.h"
#include "ecg_seg_graph.h"
#include "ecg_seg_signal.h"
#include "ecg_seg_save.h"
#include "ecg_seg_signal.h"
#include "ecg_response_def.h"
#include "ecg_seg_util.h"

static ecg_seg_graph_t g_graph_ctr = {0};

int main(int argc, char* argv[])
{
    int32_t retval = ECG_SEG_OK;
    double start_time = 0.0f, end_time = 0.0f, time_span = 0.0f;
    mat_sig_para_t input_sig_para = {.ori_l = ECG_SIGNAL_ORI_L,
                                     .k_l = ECG_SIGNAL_K_L,
                                     .padding = ECG_SIGNAL_PADDING,
                                     .stride = ECG_SIGNAL_STRIDE,};
    ree_log(LOG_DEBUG, "%s starts", __func__);
    printf("%s starts\n", __func__);
    retval = ecg_seg_graph_constructor_fp(ECG_SIGNAL_ORI_C, 
                                          ECG_MIDDLE_FEATURE_GROUP_NUM, 
                                          ECG_OUTPUT_PRED_GROUP_NUM, 
                                          ECG_CONV_RELU_FUSE_GROUP_NUM, 
                                          ECG_TRANCONV_GROUP_NUM, &g_graph_ctr);
    retval = ecg_seg_graph_input_constructor_fopen(ECG_SIGNAL,
                                                   &input_sig_para,
                                                   &g_graph_ctr);
    retval = ecg_seg_graph_context_init(&g_graph_ctr);
    start_time = now_ns();
    retval = ecg_seg_graph_forward(&g_graph_ctr);
    end_time = now_ns();
    time_span = end_time - start_time;
    printf("%s inference took %f\n", __func__, time_span);
    retval = ecg_seg_save_result((uint8_t*)ECG_PRED_LABEL,
                                 g_graph_ctr.p_out_pred->signal[0].ori_buf,
                                 2000*sizeof(uint8_t));
    ecg_seg_graph_destructor_fp(&g_graph_ctr);
    printf("%s ends retval %d\n", __func__, retval);
    ree_log(LOG_DEBUG, "%s ends retval %d", __func__, retval);
    return 0;
}
