#ifndef __ECG_SEG_SAVE_H__
#define __ECG_SEG_SAVE_H__
#include <stdint.h>
#include "arm_typedef.h"

int32_t ecg_seg_save_result(uint8_t *path,
                            void *p_buf,
                            uint32_t buf_len);

#endif
