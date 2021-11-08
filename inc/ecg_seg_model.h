#ifndef __ECG_SEG_MODEL_H__
#define __ECG_SEG_MODEL_H__

#include <stdint.h>

#include "arm_util.h"
#include "arm_typedef.h"
#include "ecg_seg_matrix.h"

#ifdef MODEL_DBG
#define MODEL_FUNC_ENTRANCE                                             FUNC_ENTANGE_LOG
#define MODEL_FUNC_EXIT                                                 FUNC_EXIT_LOG
#define MODEL_PRINTF(x...)                                              ree_printf(LOG_DEBUG, x)
#define MODEL_LOG                                                       LOG_DEBUG
#else
#define MODEL_FUNC_ENTRANCE                                             do {} while (0)
#define MODEL_FUNC_EXIT                                                 do {} while (0)
#define MODEL_PRINTF(x...)                                              do {} while (0)
#define MODEL_LOG                                                       LOG_VERBOSE
#endif

#define UNET_ENCODER1_CONV1D_BLOCK_1_CONV_WEIGHT0                       "./model_weight/unet_encoder1_conv1d_block_1_conv_weight0.bin"
#define UNET_ENCODER1_CONV1D_BLOCK_1_CONV_WEIGHT1                       "./model_weight/unet_encoder1_conv1d_block_1_conv_weight1.bin"
#define UNET_ENCODER1_CONV1D_BLOCK_1_CONV_WEIGHT2                       "./model_weight/unet_encoder1_conv1d_block_1_conv_weight2.bin"
#define UNET_ENCODER1_CONV1D_BLOCK_1_CONV_WEIGHT3                       "./model_weight/unet_encoder1_conv1d_block_1_conv_weight3.bin"

typedef struct conv_fuse_relu
{
    mat_sig_para_t weight_para;
    mat_sig_t conv_weight[4];
    mat_sig_t conv_bias[4];
} conv_fuse_relu_t;

int32_t conv_fuse_relu_constructor_fopen(mat_sig_para_t *p_para, 
                                         conv_fuse_relu_t **pp_module,
                                         char *path0, char *path1, char *path2, char *path3);

void conv_fuse_relu_destructor(conv_fuse_relu_t *p_module);

#endif
