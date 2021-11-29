#ifndef __ECG_SEG_DEF_H__
#define __ECG_SEG_DEF_H__

#define ECG_SEG_ENCODER_CONVRELU_WEIGHT_PACK_SIZE                     (12)

#define ECG_SEG_ENCODER_CONVRELU_0_K_L                                (9)
#define ECG_SEG_ENCODER_CONVRELU0_0_K_C                               (4)
#define ECG_SEG_ENCODER_CONVRELU0_1_K_C                               (16)
#define ECG_SEG_ENCODER_CONVRELU1_0_K_C                               (32)
#define ECG_SEG_ENCODER_CONVRELU1_1_K_C                               (64)
#define ECG_SEG_ENCODER_CONVRELU_0_K_DUMMING_PADDING                  (4)
#define ECG_SEG_ENCODER_CONVRELU_0_K_DUMMING_STRIDE                   (1)

#define ECG_TEST_SIGNAL_ORI_C                                         (1)
#define ECG_TEST_SIGNAL_ORI_L                                         (2000)
#define ECG_TEST_SIGNAL_MID1_ORI_L                                    (1000)
#define ECG_TEST_SIGNAL_K_L                                           (9)
#define ECG_TEST_SIGNAL_PADDING                                       (4)
#define ECG_TEST_SIGNAL_STRIDE                                        (1)
#define ECG_TEST_SIGNAL_MAX_OUT_L                                     (2000)
#define ECG_TEST_SIGNAL_MAX_K_L                                       (9)

#define ECG_TINY_TEST_SIGNAL_ORI_C                                    (1)
#define ECG_TINY_TEST_SIGNAL_ORI_L                                    (9)
#define ECG_TINY_TEST_SIGNAL_MID1_ORI_L                               (4)
#define ECG_TINY_TEST_SIGNAL_K_L                                      (9)
#define ECG_TINY_TEST_SIGNAL_PADDING                                  (4)
#define ECG_TINY_TEST_SIGNAL_STRIDE                                   (1)
#define ECG_TINY_TEST_SIGNAL_MAX_OUT_L                                (9)
#define ECG_TINY_TEST_SIGNAL_MAX_K_L                                  (9)

#define ECG_SIGNAL_ORI_C                                              (ECG_TEST_SIGNAL_ORI_C)
#define ECG_SIGNAL_ORI_L                                              (ECG_TEST_SIGNAL_ORI_L)
#define ECG_SIGNAL_K_L                                                (ECG_TEST_SIGNAL_K_L)
#define ECG_SIGNAL_PADDING                                            (ECG_TEST_SIGNAL_PADDING)
#define ECG_SIGNAL_STRIDE                                             (ECG_TEST_SIGNAL_STRIDE)
#define ECG_SIGNAL_ORI_IN_IND                                         (0)
#define ECG_SIGNAL_ORI_OUT_IND                                        (0)

#define ECG_SIGNAL_MID0_0_ORI_C                                       (4)
#define ECG_SIGNAL_MID0_ORI_L                                         (ECG_TEST_SIGNAL_ORI_L)
#define ECG_SIGNAL_MID0_K_L                                           (ECG_TEST_SIGNAL_K_L)
#define ECG_SIGNAL_MID0_PADDING                                       (ECG_TEST_SIGNAL_PADDING)
#define ECG_SIGNAL_MID0_STRIDE                                        (ECG_TEST_SIGNAL_STRIDE)
#define ECG_SIGNAL_MID0_MAX_C                                         (16)
#define ECG_SIGNAL_MID0_0_OUT_IND                                     (0)
#define ECG_SIGNAL_MID0_1_ORI_C                                       (4)
#define ECG_SIGNAL_MID0_1_IN_IND                                      (ECG_SIGNAL_MID0_0_OUT_IND)
#define ECG_SIGNAL_MID0_1_OUT_IND                                     (4)

#define ECG_SIGNAL_MAX_POOL_KERNEL_SIZE                               (8)
#define ECG_SIGNAL_MAX_POOL_STRIDE                                    (2)
#define ECG_SIGNAL_MAX_POOL_PADDING                                   (3)

#define ECG_SIGNAL_MID1_INPUT_C                                       (4)
#define ECG_SIGNAL_MID1_0_ORI_C                                       (8)
#define ECG_SIGNAL_MID1_ORI_L                                         (ECG_TEST_SIGNAL_MID1_ORI_L)
#define ECG_SIGNAL_MID1_K_L                                           (ECG_TEST_SIGNAL_K_L)
#define ECG_SIGNAL_MID1_PADDING                                       (ECG_TEST_SIGNAL_PADDING)
#define ECG_SIGNAL_MID1_STRIDE                                        (ECG_TEST_SIGNAL_STRIDE)
#define ECG_SIGNAL_MID1_MAX_C                                         (32)
#define ECG_SIGNAL_MID1_INPUT_IN_IND                                  (0)

#define ECG_SIG2COL_MAX_OUT_L                                         (ECG_TEST_SIGNAL_MAX_OUT_L)
#define ECG_SIG2COL_MAX_K_L                                           (ECG_TEST_SIGNAL_MAX_K_L)

#define ECG_MIDDLE_FEATURE_GROUP_NUM                                  (5)
#define ECG_CONV_RELU_FUSE_GROUP_NUM                                  (19)
#define ECG_OUTPUT_PRED_GROUP_NUM                                     (1)

#endif
