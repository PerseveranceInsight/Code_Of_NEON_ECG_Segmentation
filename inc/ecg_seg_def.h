#ifndef __ECG_SEG_DEF_H__
#define __ECG_SEG_DEF_H__

#define ECG_SEG_ENCODER_CONVRELU_0_K_L                                (9)
#define ECG_SEG_ENCODER_CONVRELU_0_K_C                                (4)
#define ECG_SEG_ENCODER_CONVRELU_0_K_DUMMING_PADDING                  (4)
#define ECG_SEG_ENCODER_CONVRELU_0_K_DUMMING_STRIDE                   (1)

#define ECG_TEST_SIGNAL_ORI_C                                         (1)
#define ECG_TEST_SIGNAL_ORI_L                                         (2000)
#define ECG_TEST_SIGNAL_K_L                                           (9)
#define ECG_TEST_SIGNAL_PADDING                                       (4)
#define ECG_TEST_SIGNAL_STRIDE                                        (1)
#define ECG_TEST_SIGNAL_MAX_OUT_L                                     (2000)
#define ECG_TEST_SIGNAL_MAX_K_L                                       (9)

#define ECG_TINY_TEST_SIGNAL_ORI_C                                    (1)
#define ECG_TINY_TEST_SIGNAL_ORI_L                                    (9)
#define ECG_TINY_TEST_SIGNAL_K_L                                      (9)
#define ECG_TINY_TEST_SIGNAL_PADDING                                  (4)
#define ECG_TINY_TEST_SIGNAL_STRIDE                                   (1)
#define ECG_TINY_TEST_SIGNAL_MAX_OUT_L                                (9)
#define ECG_TINY_TEST_SIGNAL_MAX_K_L                                  (9)

#define ECG_SIGNAL_ORI_C                                              (ECG_TINY_TEST_SIGNAL_ORI_C)
#define ECG_SIGNAL_ORI_L                                              (ECG_TINY_TEST_SIGNAL_ORI_L)
#define ECG_SIGNAL_K_L                                                (ECG_TINY_TEST_SIGNAL_K_L)
#define ECG_SIGNAL_PADDING                                            (ECG_TINY_TEST_SIGNAL_PADDING)
#define ECG_SIGNAL_STRIDE                                             (ECG_TINY_TEST_SIGNAL_STRIDE)

#define ECG_SIGNAL_MID1_ORI_C                                         (4)
#define ECG_SIGNAL_MID1_ORI_L                                         (ECG_TINY_TEST_SIGNAL_ORI_L)
#define ECG_SIGNAL_MID1_K_L                                           (ECG_TINY_TEST_SIGNAL_K_L)
#define ECG_SIGNAL_MID1_PADDING                                       (ECG_TINY_TEST_SIGNAL_PADDING)
#define ECG_SIGNAL_MID1_STRIDE                                        (ECG_TINY_TEST_SIGNAL_STRIDE)
#define ECG_SIGNAL_MID1_MAX_C                                         (16)

#define ECG_SIG2COL_MAX_OUT_L                                         (ECG_TINY_TEST_SIGNAL_MAX_OUT_L)
#define ECG_SIG2COL_MAX_K_L                                           (ECG_TINY_TEST_SIGNAL_MAX_K_L)

#define ECG_MIDDLE_FEATURE_GROUP_NUM                                  (5)
#define ECG_CONV_RELU_FUSE_GROUP_NUM                                  (19)
#define ECG_OUTPUT_PRED_GROUP_NUM                                     (1)

#endif
