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
    *pp_container = ree_malloc(sizeof(signal_container_t));
    ree_set(*pp_container, 0, sizeof(signal_container_t));
    (*pp_container)->signal_num = signal_num;
    ree_log(SIGNAL_LOG, "%s signal_num %d %d", __func__, (*pp_container)->signal_num, signal_num);
    (*pp_container)->signal = ree_malloc(sizeof(mat_sig_t)*signal_num);
    
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
EXIT_SIGNAL_CONTAINER_CONSTRUCTOR_FP_FOPEN:
    SIGNAL_FUNC_EXIT;
    return retval;
}

void signal_container_destructor(signal_container_t *p_container)
{
    SIGNAL_FUNC_ENTRANCE;
    for (uint32_t i = 0; i<p_container->signal_num; i++)
    {
       mat_sig_destructor(&p_container->signal[i]);
    }
    ree_free(p_container->signal);
    SIGNAL_FUNC_EXIT;
}
