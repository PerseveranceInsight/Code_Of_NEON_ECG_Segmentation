#ifndef __ECG_SEG_SIGNAL_H__
#define __ECG_SEG_SIGNAL_H__

#include <stdint.h>

#include "arm_util.h"
#include "arm_typedef.h"
#include "ecg_seg_matrix.h"

#ifdef SIGNAL_DBG
#define SIGNAL_FUNC_ENTRANCE                                            FUNC_ENTRANCE_LOG
#define SIGNAL_FUNC_EXIT                                                FUNC_EXIT_LOG
#define SIGNAL_PRINTF(x...)                                             ree_printf(LOG_DEBUG, x)
#define SIGNAL_LOG                                                      LOG_DEBUG
#else
#define SIGNAL_FUNC_ENTRANCE                                            do {} while (0)
#define SIGNAL_FUNC_EXIT                                                do {} while (0)
#define SIGNAL_PRINTF(x...)                                             do {} while (0)
#define SIGNAL_LOG                                                      LOG_VERBOSE
#endif

#define ECG_SIGNAL                                                      "./data/test_signal.bin"
#define ECG_TINY_SIGNAL                                                 "./data/test_tiny_signal.bin"

typedef struct signal_container {
    uint32_t signal_num;
    mat_sig_para_t signal_para;
    mat_sig_t *signal;
} signal_container_t;

int32_t signal_container_constructor_fp_fopen(uint32_t signal_num,
                                              mat_sig_para_t *p_para,
                                              signal_container_t **pp_container,
                                              char **signal_path);

int32_t signal_container_constructor(uint32_t signal_num,
                                     mat_sig_para_t *p_para,
                                     signal_container_t **pp_container);

void signal_container_destructor(signal_container_t *p_container);

#endif
