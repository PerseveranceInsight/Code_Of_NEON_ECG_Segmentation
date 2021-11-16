#ifndef __ECG_SEG_GEMM_H__
#define __ECG_SEG_GEMM_H__
#include <stdint.h>

#include "arm_util.h"
#include "arm_typedef.h"
#include "ecg_seg_sig2col.h"
#include "ecg_seg_signal.h"

#ifdef EN_GEMM_DBG
#define GEMM_FUNC_ENTRANCE                                              FUNC_ENTRANCE_LOG
#define GEMM_FUNC_EXIT                                                  FUNC_EXIT_LOG
#define GEMM_PRINTF(x...)                                               ree_printf(LOG_DEBUG, x)
#define GEMM_LOG                                                        LOG_DEBUG
#else
#define GEMM_FUNC_ENTRANCE                                              do {} while (0)
#define GEMM_FUNC_EXIT                                                  do {} while (0)
#define GEMM_PRINTF(x...)                                               do {} while (0)
#define GEMM_LOG                                                        LOG_VERBOSE
#endif

#define FLOAT_POINT_EXP_MAN_BIT                                         (31)

int32_t ecg_seg_fp_gemm(mat_sig_t *p_conv_weight,
                        sig2col_ctr_t *p_sig_ctr,
                        mat_sig_t *p_out_feature);

int32_t ecg_seg_fp_add_bias(mat_sig_t *p_out_feature,
                            float bias,
                            BOOL fused_relu);

#endif
