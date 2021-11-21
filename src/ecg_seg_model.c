#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arm_util.h"
#include "arm_typedef.h"
#include "ecg_seg_gemm.h"
#include "ecg_seg_matrix.h"
#include "ecg_seg_model.h"
#include "ecg_seg_sig2col.h"
#include "ecg_response_def.h"

static void printf_conv_fuse_bias(conv_fuse_relu_t *p_op)
{
    MODEL_FUNC_ENTRANCE;
    ree_check_null_exit(p_op, EXIT_PRINTF_CONV_FUSE_BIAS, "%s occurs error due to p_op is NULL", __func__);
    MODEL_PRINTF("%s %03.4f %03.4f %03.4f %03.4f\n", __func__,
                                                    p_op->conv_bias[0],
                                                    p_op->conv_bias[1],
                                                    p_op->conv_bias[2],
                                                    p_op->conv_bias[3]);
EXIT_PRINTF_CONV_FUSE_BIAS:
    MODEL_FUNC_EXIT;
}

static int32_t conv_fuse_bias_fopen(char *bias_path,
                                    float *conv_bias_buf)
{
    MODEL_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    size_t file_size = 0, read_size = 0;
    FILE *bias_file = NULL;
    ree_check_null_exit_retval(bias_path, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_BIAS_FOPEN,
                               "%s occurs error due to bias_path is NULL", __func__);
    ree_check_null_exit_retval(conv_bias_buf, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_BIAS_FOPEN,
                               "%s occurs erroe due to conv_bias_buf is NULL", __func__);
    ree_check_fopen(bias_file, bias_path, "rb", EXIT_CONV_FUSE_BIAS_FOPEN);
    ree_file_size(file_size, bias_file);
    ree_log(MODEL_LOG, "%s file_size %lu", __func__, file_size);
    ree_check_fopen(bias_file, bias_path, "rb", EXIT_CONV_FUSE_BIAS_FOPEN);
    ree_file_read(bias_file, conv_bias_buf, file_size, read_size);
EXIT_CONV_FUSE_BIAS_FOPEN:
    MODEL_FUNC_EXIT;
    return retval;
}

int32_t conv_fuse_relu_constructor_fopen(mat_sig_para_t *p_para, 
                                         conv_fuse_relu_t **pp_module,
                                         char *weight_path0, 
                                         char *weight_path1, 
                                         char *weight_path2, 
                                         char *weight_path3,
                                         char *bias_path)
{
    MODEL_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_para, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_CONSTRUCTOR,
                               "%s occurs error due to p_para is NULL", __func__);
    ree_check_null_exit_retval(pp_module, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_CONSTRUCTOR,
                               "%s occurs error due to pp_module is NULL", __func__);
    ree_check_null_exit_retval(weight_path0, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_CONSTRUCTOR,
                               "%s occurs error due to weight_path0 is NULL", __func__);
    ree_check_null_exit_retval(weight_path1, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_CONSTRUCTOR,
                               "%s occurs error due to weight_path1 is NULL", __func__);
    ree_check_null_exit_retval(weight_path2, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_CONSTRUCTOR,
                               "%s occurs error due to weight_path2 is NULL", __func__);
    ree_check_null_exit_retval(weight_path3, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_CONSTRUCTOR,
                               "%s occurs error due to weight_path3 is NULL", __func__);
    ree_check_null_exit_retval(bias_path, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_CONSTRUCTOR,
                               "%s occurs error due to bias_path is NULL", __func__);
    *pp_module = ree_malloc(sizeof(conv_fuse_relu_t));
    ree_set(*pp_module, 0, sizeof(conv_fuse_relu_t));
    retval |= mat_sig_constructor_fp_fopen(p_para,
                                           &(*pp_module)->conv_weight[0],
                                           weight_path0,
                                           TRUE);
    ree_check_true_exit((retval != ECG_SEG_OK), EXIT_CONV_FUSE_RELU_CONSTRUCTOR, "%s occurs error when constructs weight0", __func__);
    retval |= mat_sig_constructor_fp_fopen(p_para,
                                           &(*pp_module)->conv_weight[1],
                                           weight_path1,
                                           TRUE);
    ree_check_true_exit((retval != ECG_SEG_OK), EXIT_CONV_FUSE_RELU_CONSTRUCTOR, "%s occurs error when constructs weight1", __func__);
    retval |= mat_sig_constructor_fp_fopen(p_para,
                                           &(*pp_module)->conv_weight[2],
                                           weight_path2,
                                           TRUE);
    ree_check_true_exit((retval != ECG_SEG_OK), EXIT_CONV_FUSE_RELU_CONSTRUCTOR, "%s occurs error when constructs weight2", __func__);
    retval |= mat_sig_constructor_fp_fopen(p_para,
                                           &(*pp_module)->conv_weight[3],
                                           weight_path3,
                                           TRUE);
    ree_check_true_exit((retval != ECG_SEG_OK), EXIT_CONV_FUSE_RELU_CONSTRUCTOR, "%s occurs error when constructs weight3", __func__);
    conv_fuse_bias_fopen(bias_path,
                         (*pp_module)->conv_bias);
    printf_conv_fuse_bias((*pp_module));
EXIT_CONV_FUSE_RELU_CONSTRUCTOR:
    MODEL_FUNC_EXIT;
    return retval;
}

int32_t conv_fuse_relu_constructor_static(uint32_t conv_fuse_relu_num,
                                          mat_sig_para_t *p_para,
                                          conv_fuse_relu_t **pp_module,
                                          void **pp_weight_buf,
                                          void **pp_bias_buf)
{
    MODEL_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_para, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_CONSTRUCTOR_STATIC,
                               "%s occurs error due to p_para is NULL", __func__);
    ree_check_null_exit_retval(pp_module, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_CONSTRUCTOR_STATIC,
                               "%s occurs error due to pp_module is NULL", __func__);
    ree_check_null_exit_retval(pp_weight_buf, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_CONSTRUCTOR_STATIC,
                               "%s occurs error due to pp_weight is NULL", __func__);
    ree_check_null_exit_retval(pp_bias_buf, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_CONSTRUCTOR_STATIC,
                               "%s occurs error due to  pp_bias_buf is NULL", __func__);
    if (!(*pp_module))
    {
        ree_log(MODEL_LOG, "%s prepares to allocate pp_module", __func__);
        *pp_module = ree_malloc(sizeof(conv_fuse_relu_t));
        ree_set(*pp_module, 0, sizeof(conv_fuse_relu_t));
    } else
    {
        ree_log(MODEL_LOG, "%s has already allocated pp_module", __func__);
    }

EXIT_CONV_FUSE_RELU_CONSTRUCTOR_STATIC:
    MODEL_FUNC_EXIT;
    return retval;
}

int32_t conv_fuse_relu_forward(conv_fuse_relu_t *p_module,
                               sig2col_ctr_t *p_col_ctr,
                               signal_container_t *p_in_sig_con,
                               signal_container_t *p_out_sig_con,
                               uint32_t input_num,
                               uint32_t output_num)
{
    MODEL_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_module, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_FORWARD,
                               "%s occurs error due to p_module is NULL", __func__);
    ree_check_null_exit_retval(p_col_ctr, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_FORWARD,
                               "%s occurs error due to p_col_ctr is NULL", __func__);
    ree_check_null_exit_retval(p_in_sig_con, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_FORWARD,
                               "%s occurs error due to p_in_sig_con is NULL", __func__);
    ree_check_null_exit_retval(p_out_sig_con, retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_FORWARD,
                               "%s occurs error due to p_out_sig_con is NULL", __func__);
    ree_check_true_exit_retval((input_num == 0), retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_FORWARD,
                               "%s directly return due to input_num == 0", __func__);
    ree_check_true_exit_retval((output_num == 0), retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_FORWARD,
                               "%s directly return due to output_num == 0", __func__);
    ree_log(MODEL_LOG, "%s input_num %d output_num %d", __func__, input_num, output_num);

    for (uint32_t in_ind = 0; in_ind<input_num; in_ind++)
    {
        ree_check_null_exit_retval((&(p_in_sig_con->signal[in_ind])), retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_FORWARD,
                                   "%s occurs error due to p_in_sig_con->signal[in_ind] is NULL", __func__);
        retval = sig2col_mat_fp(p_col_ctr, &(p_in_sig_con->signal[in_ind]));
        ree_check_true_exit((retval != ECG_SEG_OK), EXIT_CONV_FUSE_RELU_FORWARD,
                             "%s occurs error due to sig2col_mat_fp of in_ind %d failed", __func__, in_ind);
        for (uint32_t out_ind = 0; out_ind<output_num; out_ind++)
        {
            ree_check_null_exit_retval(&(p_out_sig_con->signal[out_ind]), retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_FORWARD,
                                       "%s occurs error due to p_out_sig_con->signal[out_ind] is NULL out_ind %d", __func__, out_ind);
            retval = ecg_seg_fp_gemm(&(p_module->conv_weight[out_ind]),
                                     p_col_ctr,
                                     &(p_out_sig_con->signal[out_ind]));
            ree_check_true_exit((retval != ECG_SEG_OK), EXIT_CONV_FUSE_RELU_FORWARD,
                                 "%s occurs error due to ecg_seg_fp_gemm of in_ind %d out_ind %d failed", __func__, in_ind, out_ind);
        }
    }

    for (uint32_t out_ind = 0; out_ind<output_num; out_ind++)
    {
        ree_check_null_exit_retval(&(p_out_sig_con->signal[out_ind]), retval, ECG_SEG_INVALID_PARAM, EXIT_CONV_FUSE_RELU_FORWARD,
                                   "%s occurs error due to p_out_sig_con->signal[out_ind] is NULL out_ind %d", __func__, out_ind);
        retval = ecg_seg_fp_add_bias((&p_out_sig_con->signal[out_ind]),
                                     p_module->conv_bias[out_ind],
                                     TRUE);
        print_mat_ori_fp(&(p_out_sig_con->signal[out_ind]));
        ree_check_true_exit((retval != ECG_SEG_OK), EXIT_CONV_FUSE_RELU_FORWARD,
                                         "%s occurs error due to ecg_seg_fp_add_bias of out_ind %d failed", __func__, out_ind);
    }

EXIT_CONV_FUSE_RELU_FORWARD:
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

