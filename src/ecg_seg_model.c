#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arm_util.h"
#include "arm_typedef.h"
#include "ecg_seg_matrix.h"
#include "ecg_seg_model.h"
#include "ecg_response_def.h"

int32_t conv_fuse_relu_constructor_fopen(mat_sig_para_t *p_para, 
                                         conv_fuse_relu_t **pp_module,
                                         char *path0, char *path1, char *path2, char *path3)
{
    MODEL_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_para, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_CONSTRUCTOR,
                               "%s occurs error due to p_para is NULL", __func__);
    ree_check_null_exit_retval(pp_module, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_CONSTRUCTOR,
                               "%s occurs error due to pp_module is NULL", __func__);
    *pp_module = ree_malloc(sizeof(conv_fuse_relu_t));
    ree_set(*pp_module, 0, sizeof(conv_fuse_relu_t));
    retval |= mat_sig_constructor_fp_fopen(p_para,
                                           &(*pp_module)->conv_weight[0],
                                           path0,
                                           TRUE);
    ree_check_true_exit((retval != ECG_SEG_OK), EXIT_CONV_FUSE_RELU_CONSTRUCTOR, "%s occurs error when constructs weight0", __func__);
    retval |= mat_sig_constructor_fp_fopen(p_para,
                                           &(*pp_module)->conv_weight[1],
                                           path1,
                                           TRUE);
    ree_check_true_exit((retval != ECG_SEG_OK), EXIT_CONV_FUSE_RELU_CONSTRUCTOR, "%s occurs error when constructs weight1", __func__);
    retval |= mat_sig_constructor_fp_fopen(p_para,
                                           &(*pp_module)->conv_weight[2],
                                           path2,
                                           TRUE);
    ree_check_true_exit((retval != ECG_SEG_OK), EXIT_CONV_FUSE_RELU_CONSTRUCTOR, "%s occurs error when constructs weight2", __func__);
    retval |= mat_sig_constructor_fp_fopen(p_para,
                                           &(*pp_module)->conv_weight[3],
                                           path3,
                                           TRUE);
    ree_check_true_exit((retval != ECG_SEG_OK), EXIT_CONV_FUSE_RELU_CONSTRUCTOR, "%s occurs error when constructs weight3", __func__);
EXIT_CONV_FUSE_RELU_CONSTRUCTOR:
    MODEL_FUNC_EXIT;
    return retval;
}

void conv_fuse_relu_destructor(conv_fuse_relu_t *p_module)
{
    MODEL_FUNC_ENTRANCE;
    ree_check_null_exit(p_module, EXIT_CONV_FUSE_RELU_DESTRUCTOR, "%s directly return due to p_module is NULL", __func__);
    for (uint32_t i = 0; i<4; i++)
    {
        mat_sig_destructor(&p_module->conv_weight[i]);
    }
EXIT_CONV_FUSE_RELU_DESTRUCTOR:
    MODEL_FUNC_EXIT;
}
