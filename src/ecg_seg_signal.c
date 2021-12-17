#include <stdlib.h>
#include <string.h>

#include "arm_util.h"
#include "arm_typedef.h"
#include "ecg_response_def.h"
#include "ecg_seg_matrix.h"
#include "ecg_seg_signal.h"

int32_t signal_container_constructor_fp_fopen(uint32_t signal_num,
                                              mat_sig_para_t *p_para,
                                              signal_container_t **pp_container,
                                              char **signal_path)
{
    SIGNAL_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_para, retval, ECG_SEG_INVALID_PARAM, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP_FOPEN,
                               "%s occurs error due to p_para is NULL", __func__);
    ree_check_null_exit_retval(pp_container, retval, ECG_SEG_INVALID_PARAM, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP_FOPEN,
                               "%s occurs error due to pp_container is NULL", __func__);
    ree_check_null_exit_retval(signal_path, retval, ECG_SEG_INVALID_PARAM, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP_FOPEN,
                               "%s occurs error due to signal_path is NULL", __func__);
    ree_check_true_exit((signal_num == 0), EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP_FOPEN, "%s directly returns due to signal_num == 0", __func__);

    if (!(*pp_container))
    {
        ree_log(SIGNAL_LOG, "%s prepares to allocate pp_container", __func__);
        *pp_container = ree_malloc(sizeof(signal_container_t));
        ree_check_null_exit_retval(*pp_container, retval, ECG_SEG_ALLOC_FAILED, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP_FOPEN,
                                   "%s occurs error due to pp_container is NULL", __func__);
        ree_set(*pp_container, 0, sizeof(signal_container_t));
    } else 
    {
        ree_log(SIGNAL_LOG, "%s pp_container has already allocated", __func__);
    }
    (*pp_container)->signal_num = signal_num;
    ree_log(SIGNAL_LOG, "%s signal_num %d %d", __func__, (*pp_container)->signal_num, signal_num);
    (*pp_container)->signal = ree_malloc(sizeof(mat_sig_t)*signal_num);
    ree_check_null_exit_retval((*pp_container)->signal, retval, ECG_SEG_ALLOC_FAILED, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP_FOPEN,
                               "%s occurs error due to allocate (*pp_container)->signal faild", __func__);
     ree_set((*pp_container)->signal, 0, sizeof(mat_sig_t)*signal_num);

    for (uint32_t i=0; i<signal_num; i++)
    {
        ree_log(SIGNAL_LOG, "%s signal index i %d", __func__, i);
        ree_check_null_exit_retval(signal_path[i], retval, ECG_SEG_INVALID_PARAM, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP_FOPEN,
                                   "%s occurs error due to signal_path[i] is NULL", __func__);
        ree_check_null_exit_retval(&(*pp_container)->signal[i], retval, ECG_SEG_ALLOC_FAILED, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP_FOPEN,
                                   "%s occurs error due to (*pp_container)->signal[i] is NULL", __func__);
        ree_log(SIGNAL_LOG, "%s signal_path %s", __func__, signal_path[i]);
        mat_sig_constructor_fp_fopen(p_para, &(*pp_container)->signal[i], signal_path[i], FALSE);
    }
    (*pp_container)->inited = TRUE;
EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP_FOPEN:
    SIGNAL_FUNC_EXIT;
    return retval;
}

int32_t signal_container_constructor_fp(uint32_t signal_num, mat_sig_para_t *p_para, signal_container_t **pp_container)
{
    SIGNAL_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_para, retval, ECG_SEG_INVALID_PARAM, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP,
                               "%s occurs error due to p_para is NULL", __func__);
    ree_check_null_exit_retval(pp_container, retval, ECG_SEG_INVALID_PARAM, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP,
                               "%s occurs error due to pp_container is NULL", __func__);
    ree_check_true_exit((signal_num == 0), EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP, "%s directly returns due to signal_num == 0", __func__);

    if (!(*pp_container))
    {
        ree_log(SIGNAL_LOG, "%s prepares to allocated pp_container", __func__);
        *pp_container = ree_malloc(sizeof(signal_container_t));
        ree_check_null_exit_retval(*pp_container, retval, ECG_SEG_ALLOC_FAILED, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP,
                                   "%s occurs error due to pp_container is NULL", __func__);
        ree_set(*pp_container, 0, sizeof(signal_container_t));
    } else 
    {
        ree_log(SIGNAL_LOG, "%s pp_container has already allocated", __func__);
    }
    (*pp_container)->signal_num = signal_num;
    ree_log(SIGNAL_LOG, "%s signal_num %d %d", __func__, (*pp_container)->signal_num, signal_num);
    (*pp_container)->signal = ree_malloc(sizeof(mat_sig_t)*signal_num);
    ree_check_null_exit_retval((*pp_container)->signal, retval, ECG_SEG_ALLOC_FAILED, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP,
                               "%s occurs error due to allocate (*pp_container)->signal faild", __func__);
    ree_set((*pp_container)->signal, 0, sizeof(mat_sig_t)*signal_num);
    ree_cpy((void*)&((*pp_container)->signal_para), p_para, sizeof(mat_sig_para_t));

    for (uint32_t i=0; i<signal_num; i++)
    {
        ree_log(SIGNAL_LOG, "%s signal index i %d", __func__, i);
        ree_check_null_exit_retval(&((*pp_container)->signal[i]), retval, ECG_SEG_ALLOC_FAILED, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP,
                                   "%s occurs error due to (*pp_container)->signal[i] is NULL", __func__);
        mat_sig_constructor_fp(p_para, &(*pp_container)->signal[i], FALSE);
    }
    (*pp_container)->inited = TRUE;
EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP:
    SIGNAL_FUNC_EXIT;
    return retval;
}

int32_t signal_container_constructor_uint8(uint32_t signal_num, mat_sig_para_t *p_para, signal_container_t **pp_container)
{
    SIGNAL_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    ree_check_null_exit_retval(p_para, retval, ECG_SEG_INVALID_PARAM, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_UINT8,
                               "%s occurs error due to p_para is NULL", __func__);
    ree_check_null_exit_retval(pp_container, retval, ECG_SEG_INVALID_PARAM, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_UINT8,
                               "%s occurs error due to pp_container is NULL", __func__);
    ree_check_true_exit((signal_num == 0), EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_UINT8, "%s directly returns due to signal_num == 0", __func__);

    if (!(*pp_container))
    {
        ree_log(SIGNAL_LOG, "%s prepares to allocated pp_container", __func__);
        *pp_container = ree_malloc(sizeof(signal_container_t));
        ree_check_null_exit_retval(*pp_container, retval, ECG_SEG_ALLOC_FAILED, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_UINT8,
                                   "%s occurs error due to pp_container is NULL", __func__);
        ree_set(*pp_container, 0, sizeof(signal_container_t));
    } else 
    {
        ree_log(SIGNAL_LOG, "%s pp_container has already allocated", __func__);
    }
    (*pp_container)->signal_num = signal_num;
    ree_log(SIGNAL_LOG, "%s signal_num %d %d", __func__, (*pp_container)->signal_num, signal_num);
    (*pp_container)->signal = ree_malloc(sizeof(mat_sig_t)*signal_num);
    ree_check_null_exit_retval((*pp_container)->signal, retval, ECG_SEG_ALLOC_FAILED, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_UINT8,
                               "%s occurs error due to allocate (*pp_container)->signal faild", __func__);
    ree_set((*pp_container)->signal, 0, sizeof(mat_sig_t)*signal_num);
    ree_cpy((void*)&((*pp_container)->signal_para), p_para, sizeof(mat_sig_para_t));

    for (uint32_t i=0; i<signal_num; i++)
    {
        ree_log(SIGNAL_LOG, "%s signal index i %d", __func__, i);
        ree_check_null_exit_retval(&((*pp_container)->signal[i]), retval, ECG_SEG_ALLOC_FAILED, EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_UINT8,
                                   "%s occurs error due to (*pp_container)->signal[i] is NULL", __func__);
        mat_sig_constructor_uint8(p_para, &(*pp_container)->signal[i], FALSE);
    }
    (*pp_container)->inited = TRUE;
EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_UINT8:
    SIGNAL_FUNC_EXIT;
    return retval;
}

int32_t signal_container_reset_fp(signal_container_t *p_container,
                                  uint32_t reset_num,
                                  uint32_t reset_start_ind)
{
    SIGNAL_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    uint32_t res_start_ind = reset_start_ind;
    uint32_t res_end_ind = reset_start_ind + reset_num;
    ree_check_null_exit_retval(p_container, retval, ECG_SEG_INVALID_PARAM, EXIT_SIGNAL_CONTAINER_RESET_FP,
                               "%s occurs error due to p_container is NULL", __func__);
    ree_log(SIGNAL_LOG, "%s reset_start_ind %d reset_num %d", __func__, reset_start_ind, reset_num);
    ree_log(SIGNAL_LOG, "%s res_start_ind %d res_end_ind %d", __func__, res_start_ind, res_end_ind);
    ree_check_true_exit_retval((res_start_ind > p_container->signal_num), retval, ECG_SEG_INVALID_PARAM, EXIT_SIGNAL_CONTAINER_RESET_FP,
                               "%s occurs error due to res_start_ind > p_container->signal_num %d", __func__, p_container->signal_num);
    ree_check_true_exit_retval((res_end_ind > p_container->signal_num), retval, ECG_SEG_INVALID_PARAM, EXIT_SIGNAL_CONTAINER_RESET_FP,
                               "%s occurs error due to res_end_ind > p_container->signal_num %d", __func__, p_container->signal_num);
    
    for (uint32_t res_ind = res_start_ind; res_ind < res_end_ind; res_ind++)
    {
        ree_log(SIGNAL_LOG, "%s res_ind %d", __func__, res_ind);
        ree_check_null_exit_retval((&p_container->signal[res_ind]), retval, ECG_SEG_ALLOC_FAILED, EXIT_SIGNAL_CONTAINER_RESET_FP,
                                   "%s occurs error due to p_container->signal[res_ind] is NULL", __func__);
        retval = mat_sig_reset_fp((&p_container->signal[res_ind]));
        ree_check_true_exit((retval != ECG_SEG_OK), 
                            EXIT_SIGNAL_CONTAINER_RESET_FP, 
                            "%s occurs error due to retval of mat_sig_reset_fp != ECG_SEG_OK", __func__);
    }

EXIT_SIGNAL_CONTAINER_RESET_FP:
    SIGNAL_FUNC_EXIT;
    return retval;
}

int32_t signal_container_reset_uint8(signal_container_t *p_container,
                                     uint32_t reset_num,
                                     uint32_t reset_start_ind)
{
    SIGNAL_FUNC_ENTRANCE;
    int32_t retval = ECG_SEG_OK;
    uint32_t res_start_ind = reset_start_ind;
    uint32_t res_end_ind = reset_start_ind + reset_num;
    ree_check_null_exit_retval(p_container, retval, ECG_SEG_INVALID_PARAM, EXIT_SIGNAL_CONTAINER_RESET_UINT8,
                               "%s occurs error due to p_container is NULL", __func__);
    ree_log(SIGNAL_LOG, "%s reset_start_ind %d reset_num %d", __func__, reset_start_ind, reset_num);
    ree_log(SIGNAL_LOG, "%s res_start_ind %d res_end_ind %d", __func__, res_start_ind, res_end_ind);
    ree_check_true_exit_retval((res_start_ind > p_container->signal_num), retval, ECG_SEG_INVALID_PARAM, EXIT_SIGNAL_CONTAINER_RESET_UINT8,
                               "%s occurs error due to res_start_ind > p_container->signal_num %d", __func__, p_container->signal_num);
    ree_check_true_exit_retval((res_end_ind > p_container->signal_num), retval, ECG_SEG_INVALID_PARAM, EXIT_SIGNAL_CONTAINER_RESET_UINT8,
                               "%s occurs error due to res_end_ind > p_container->signal_num %d", __func__, p_container->signal_num);
    
    for (uint32_t res_ind = res_start_ind; res_ind < res_end_ind; res_ind++)
    {
        ree_log(SIGNAL_LOG, "%s res_ind %d", __func__, res_ind);
        ree_check_null_exit_retval((&p_container->signal[res_ind]), retval, ECG_SEG_ALLOC_FAILED, EXIT_SIGNAL_CONTAINER_RESET_UINT8,
                                   "%s occurs error due to p_container->signal[res_ind] is NULL", __func__);
        retval = mat_sig_reset_uint8((&p_container->signal[res_ind]));
        ree_check_true_exit((retval != ECG_SEG_OK), 
                            EXIT_SIGNAL_CONTAINER_RESET_UINT8, 
                            "%s occurs error due to retval of mat_sig_reset_fp != ECG_SEG_OK", __func__);
    }

EXIT_SIGNAL_CONTAINER_RESET_UINT8:
    SIGNAL_FUNC_EXIT;
    return retval;
}

void signal_container_destructor(signal_container_t *p_container)
{
    SIGNAL_FUNC_ENTRANCE;
    if (p_container->signal)
    {
        for (uint32_t i = 0; i<p_container->signal_num; i++)
        {
            mat_sig_destructor(&(p_container->signal[i]));
        }
    }
    ree_free(p_container->signal);
    SIGNAL_FUNC_EXIT;
}
