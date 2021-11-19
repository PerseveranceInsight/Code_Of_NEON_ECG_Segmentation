#ifndef __ECG_SEG_RESPONSE_DEF_H__
#define __ECG_SEG_RESPONSE_DEF_H__

typedef enum ecg_seg_res {
    ECG_SEG_OK = 0,
    ECG_SEG_INVALID_PARAM = 1,
    ECG_SEG_ALLOC_FAILED = 2,
    ECG_SEG_ERROR_STATE = 3,
} ecg_seg_res_t;

#endif
