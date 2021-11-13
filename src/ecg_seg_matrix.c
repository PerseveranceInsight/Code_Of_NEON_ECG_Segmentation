#include <stdlib.h>
#include <string.h>

#include "arm_util.h"
#include "arm_typedef.h"
#include "ecg_response_def.h"
#include "ecg_seg_matrix.h"

static void mat_sig_fp_fopen(void *p_mat, char *path)
{
    MATRIX_FUNC_ENTRANCE;
    size_t file_size = 0, read_size;
    FILE *mat_file = NULL;
    ree_check_null_exit(p_mat, EXIT_MAT_SIG_FP_FOPEN, "%s occurs error due to p_mat is NULL", __func__);
    ree_check_null_exit(path, EXIT_MAT_SIG_FP_FOPEN, "%s occurs error due to path is NULL", __func__);
    ree_check_fopen(mat_file, path, "rb", EXIT_MAT_SIG_FP_FOPEN);
    ree_file_size(file_size, mat_file);
    ree_log(MATRIX_LOG, "%s file_size %lu", __func__, file_size);
    ree_check_fopen(mat_file, path, "rb", EXIT_MAT_SIG_FP_FOPEN);
    ree_file_read(mat_file, p_mat, file_size, read_size);
EXIT_MAT_SIG_FP_FOPEN:
    MATRIX_FUNC_EXIT;
}

void print_mat_sig_para(mat_sig_para_t *p_param)
{
    MATRIX_FUNC_ENTRANCE;
    MATRIX_PRINTF("%s p_param->ori_l %d\n", __func__, p_param->ori_l);
    MATRIX_PRINTF("%s p_param->k_l %d\n", __func__, p_param->k_l);
    MATRIX_PRINTF("%s p_param->padding %d\n", __func__, p_param->padding);
    MATRIX_PRINTF("%s p_param->stride %d\n", __func__, p_param->stride);
    MATRIX_FUNC_EXIT;
}

void print_mat_para(mat_sig_t *p_mat)
{
    MATRIX_FUNC_ENTRANCE;
    MATRIX_PRINTF("%s p_mat->ori_l %d\n", __func__, p_mat->ori_l);
    MATRIX_PRINTF("%s p_mat->out_l %d\n", __func__, p_mat->out_l);
    MATRIX_PRINTF("%s p_mat->col_h %d\n", __func__, p_mat->col_h);
    MATRIX_PRINTF("%s p_mat->col_w %d\n", __func__, p_mat->col_w);
    MATRIX_PRINTF("%s p_mat->pack_w_step %d\n", __func__, p_mat->pack_w_step);
    MATRIX_PRINTF("%s p_mat->pack_h %d\n", __func__, p_mat->pack_h);
    MATRIX_PRINTF("%s p_mat->pack_w %d\n", __func__, p_mat->pack_w);
    MATRIX_PRINTF("%s p_mat->pack_ele %d\n", __func__, p_mat->pack_ele);
    MATRIX_PRINTF("%s p_mat->padding %d\n", __func__, p_mat->padding);
    MATRIX_PRINTF("%s p_mat->stride %d\n", __func__, p_mat->stride);
    MATRIX_FUNC_EXIT;
}

void print_mat_ori_fp(mat_sig_t *p_mat)
{
    MATRIX_FUNC_ENTRANCE;
    ree_check_null_exit(p_mat, EXIT_PRINT_MAT_ORI_FP, "%s occurs error due to p_mat is NULL", __func__);
    float *p_ele = NULL;
    MATRIX_PRINTF("%s p_mat->ori_l %d\n", __func__, p_mat->ori_l);
    ree_check_null_exit(p_mat->ori_buf, EXIT_PRINT_MAT_ORI_FP, "%s occurs error due to p_mat->ori_buf is NULL", __func__);
    p_ele = p_mat->ori_buf;
    for (uint32_t i = 0; i<p_mat->ori_l; i++)
    {
        MATRIX_PRINTF("%3.05f ", *p_ele);
        p_ele++;
    }
    MATRIX_PRINTF("\n");
EXIT_PRINT_MAT_ORI_FP:
    MATRIX_FUNC_EXIT;
}

int32_t mat_sig_constructor_fp_fopen(mat_sig_para_t *p_param,
                                     mat_sig_t *p_mat,
                                     char *path,
                                     BOOL kernel)
{
    MATRIX_FUNC_ENTRANCE;
    uint32_t ele_num = 0;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_param, retval, ECG_SEG_INVALID_PARAM, EXIT_MAT_SIG_CONSTRUCTOR_FP_FOPEN,
                               "%s occurs error due to p_param is NULL", __func__);
    ree_check_null_exit_retval(p_mat, retval, ECG_SEG_INVALID_PARAM, EXIT_MAT_SIG_CONSTRUCTOR_FP_FOPEN,
                               "%s occurs error due to p_mat is NULL", __func__);
    ree_check_null_exit_retval(path, retval, ECG_SEG_INVALID_PARAM, EXIT_MAT_SIG_CONSTRUCTOR_FP_FOPEN,
                               "%s occurs error due to path is NULL", __func__);
    ree_log(MATRIX_LOG, "%s kernel %d", __func__, kernel);
    print_mat_sig_para(p_param);

    if (kernel)
    {
        p_mat->ori_l = p_param->ori_l;
        p_mat->out_l = p_param->ori_l;
        p_mat->pack_w_step = p_mat->out_l / FP_PACK_SIZE + 1;
        p_mat->pack_h = 1;
    } else
    {
        p_mat->ori_l = p_param->ori_l;
        p_mat->out_l = (p_param->ori_l+2*p_param->padding-p_param->k_l)/p_param->stride+1;
        p_mat->col_h = p_param->k_l;
        p_mat->col_w = p_mat->out_l;
        p_mat->pack_w_step = p_mat->out_l / FP_PACK_SIZE + 1;
        p_mat->pack_h = p_mat->col_h; 
    }
    p_mat->pack_w = p_mat->pack_w_step * FP_PACK_SIZE;
    p_mat->pack_ele = p_mat->pack_h * p_mat->pack_w;
    p_mat->padding = p_param->padding;
    p_mat->stride = p_param->stride;
    print_mat_para(p_mat); 
    if (kernel)
    {
        ele_num = p_mat->pack_ele;
        ree_log(MATRIX_LOG, "%s prepares to allocate kernel buffer", __func__);
    } else
    {
        ele_num = (p_mat->ori_l/FP_PACK_SIZE + 1)*FP_PACK_SIZE;
        ree_log(MATRIX_LOG, "%s prepares to allocate signal buffer", __func__);
    }
    ree_log(MATRIX_LOG, "%s prepares to allocate %d * ELE_FP_SIZE", __func__, ele_num);
    p_mat->ori_buf = ree_malloc(ele_num*ELE_FP_SIZE);
    ree_check_null_exit_retval(p_mat->ori_buf, retval, ECG_SEG_ALLOC_FAILED, EXIT_MAT_SIG_CONSTRUCTOR_FP_FOPEN,
                               "%s occurs error due to allocate p_mat->ori_buf failed", __func__);
    ree_set(p_mat->ori_buf, 0, ele_num*ELE_FP_SIZE);
    mat_sig_fp_fopen(p_mat->ori_buf, path);
    print_mat_ori_fp(p_mat);
EXIT_MAT_SIG_CONSTRUCTOR_FP_FOPEN:
    MATRIX_FUNC_EXIT;
    return retval;
}

void mat_sig_destructor(mat_sig_t *p_mat)
{
    MATRIX_FUNC_ENTRANCE;
    ree_check_null_exit(p_mat, EXIT_MAT_SIG_DESTRUCTOR, "%s directly returns due to p_mat is NULL", __func__);
    ree_free(p_mat->ori_buf);
EXIT_MAT_SIG_DESTRUCTOR:
    MATRIX_FUNC_EXIT;
}
