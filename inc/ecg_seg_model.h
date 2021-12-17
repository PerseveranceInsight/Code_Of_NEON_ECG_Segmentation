#ifndef __ECG_SEG_MODEL_H__
#define __ECG_SEG_MODEL_H__

#include <stdint.h>

#include "arm_util.h"
#include "arm_typedef.h"
#include "ecg_seg_matrix.h"
#include "ecg_seg_signal.h"
#include "ecg_seg_sig2col.h"

#ifdef MODEL_DBG
#define MODEL_FUNC_ENTRANCE                                             FUNC_ENTRANCE_LOG
#define MODEL_FUNC_EXIT                                                 FUNC_EXIT_LOG
#define MODEL_PRINTF(x...)                                              ree_printf(LOG_DEBUG, x)
#define MODEL_LOG                                                       LOG_DEBUG
#else
#define MODEL_FUNC_ENTRANCE                                             do {} while (0)
#define MODEL_FUNC_EXIT                                                 do {} while (0)
#define MODEL_PRINTF(x...)                                              do {} while (0)
#define MODEL_LOG                                                       LOG_VERBOSE
#endif

typedef struct conv_fuse_relu
{
    BOOL inited;
    uint32_t conv_fuse_relu_c;
    mat_sig_para_t weight_para;
    mat_sig_t *conv_weight;
    float *conv_bias;
} conv_fuse_relu_t;

typedef struct max_pool_parameters
{
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;
} max_pool_parameters_t;

int32_t conv_fuse_relu_constructor_fopen(mat_sig_para_t *p_para, 
                                         conv_fuse_relu_t **pp_module,
                                         char *weight_path0, 
                                         char *weight_path1, 
                                         char *weight_path2, 
                                         char *weight_path3,
                                         char *bias_path);

int32_t conv_fuse_relu_constructor_static(uint32_t conv_fuse_relu_c,
                                          mat_sig_para_t *p_para,
                                          conv_fuse_relu_t **pp_module,
                                          void **pp_weight_buf,
                                          void **pp_bias_buf);

int32_t conv_fuse_relu_forward(conv_fuse_relu_t *p_module,
                               sig2col_ctr_t *p_col_ctr,
                               signal_container_t *p_in_sig_con,
                               signal_container_t *p_out_sig_con,
                               uint32_t input_num,
                               uint32_t input_start_ind,
                               uint32_t output_num,
                               uint32_t output_start_ind);

int32_t max_pool_forward(max_pool_parameters_t *p_parameters,
                         signal_container_t *p_in_sig_con,
                         signal_container_t *p_out_sig_con,
                         uint32_t input_num,
                         uint32_t input_start_ind,
                         uint32_t output_num,
                         uint32_t output_start_ind);

int32_t tranconv_forward(conv_fuse_relu_t *p_tran_module,
                         sig2col_ctr_t *p_col_ctr,
                         signal_container_t *p_in_sig_con,
                         signal_container_t *p_out_sig_con,
                         mat_sig_tran_conv_para_t *p_tran_conv_para, 
                         uint32_t input_num,
                         uint32_t input_start_ind,
                         uint32_t output_num, 
                         uint32_t output_start_ind);

int32_t decoder_conv_fuse_relu_forward(conv_fuse_relu_t *p_module,
                                       sig2col_ctr_t *p_col_ctr,
                                       signal_container_t *p_in_sig_con,
                                       signal_container_t *p_out_sig_con,
                                       mat_decoder_conv_para_t *p_decoder_para,
                                       uint32_t input_num,
                                       uint32_t input_start_ind,
                                       uint32_t output_num,
                                       uint32_t output_start_ind);

int32_t argmax_forward(signal_container_t *p_in_sig_con,
                       signal_container_t *p_out_isg_con,
                       uint32_t input_num,
                       uint32_t input_start_ind,
                       uint32_t output_start_ind); 

void conv_fuse_relu_destructor(conv_fuse_relu_t *p_module);

void conv_fuse_relu_destructor_static(conv_fuse_relu_t *p_module);

#endif
