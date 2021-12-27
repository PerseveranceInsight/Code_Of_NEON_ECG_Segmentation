#include <stdint.h>
#include <stdio.h>
#include "arm_util.h"
#include "arm_typedef.h"
#include "ecg_response_def.h"
#include "ecg_seg_save.h"

int32_t ecg_seg_save_result(uint8_t *path, void *p_buf, uint32_t buf_len)
{
    FUNC_ENTRANCE_LOG;
    int32_t retval = ECG_SEG_OK;
    FILE *result_file = NULL;
    ree_check_null_exit_retval(path, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_SAVE_RESULT,
                          "%s occurs error due to path is NULL", __func__);
    ree_check_null_exit_retval(p_buf, retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_SAVE_RESULT,
                               "%s occurs error due to p_buf is NULL", __func__);
    ree_check_true_exit_retval((buf_len == 0), retval, ECG_SEG_INVALID_PARAM, EXIT_ECG_SEG_SAVE_RESULT,
                               "%s directly return due to buf_len==0", __func__);
    ree_check_fopen(result_file, (char*)path, "wb", EXIT_ECG_SEG_SAVE_RESULT);
    ree_file_write(result_file, p_buf, (char*)path, buf_len);
EXIT_ECG_SEG_SAVE_RESULT:
    FUNC_EXIT_LOG;
    return retval;
}
