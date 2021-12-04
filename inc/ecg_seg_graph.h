#ifndef __ECG_SEG_GRAPH_H__
#define __ECG_SEG_GRAPH_H__
#include <stdint.h>

#include "arm_typedef.h"
#include "arm_util.h"
#include "ecg_seg_def.h"
#include "ecg_seg_matrix.h"
#include "ecg_seg_model.h"
#include "ecg_seg_sig2col.h"
#include "ecg_seg_signal.h"

#ifdef EN_GRAPH_DBG
#define GRAPH_FUNC_ENTRANCE                                              FUNC_ENTRANCE_LOG
#define GRAPH_FUNC_EXIT                                                  FUNC_EXIT_LOG
#define GRAPH_PRINTF(x...)                                               ree_printf(LOG_DEBUG, x)
#define GRAPH_LOG                                                        LOG_DEBUG
#else
#define GRAPH_FUNC_ENTRANCE                                              do {} while (0)
#define GRAPH_FUNC_EXIT                                                  do {} while (0)
#define GRAPH_PRINTF(x...)                                               do {} while (0)
#define GRAPH_LOG                                                        LOG_VERBOSE
#endif

typedef struct ecg_seg_graph
{
    BOOL inited;
    uint32_t in_num;
    uint32_t mid_num;
    uint32_t out_num;
    uint32_t conv_fuse_relu_num;
    uint32_t tranconv_num;
    conv_fuse_relu_t *p_modules;
    conv_fuse_relu_t *p_tranconv_modules;
    signal_container_t *p_in_sigs;
    signal_container_t *p_mid_features;
    signal_container_t *p_out_pred;
    sig2col_ctr_t *p_sig2col_ctr;
} ecg_seg_graph_t;

int32_t ecg_seg_graph_constructor_fp(uint32_t in_num, 
                                     uint32_t mid_num,
                                     uint32_t out_num,
                                     uint32_t conv_fuse_relu_num,
                                     uint32_t tranconv_num,
                                     ecg_seg_graph_t *p_graph);

int32_t ecg_seg_graph_input_constructor_fopen(char *p_sig_path,
                                              mat_sig_para_t *p_sig_para,
                                              ecg_seg_graph_t *p_graph);

int32_t ecg_seg_graph_context_init(ecg_seg_graph_t *p_graph);

int32_t ecg_seg_graph_forward(ecg_seg_graph_t *p_graph);

int32_t ecg_seg_graph_destructor_fp(ecg_seg_graph_t *p_graph);

#endif
