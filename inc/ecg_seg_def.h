#ifndef __ECG_SEG_DEF_H__
#define __ECG_SEG_DEF_H__

#define ECG_SEG_ENCODER_CONVRELU_WEIGHT_PACK_SIZE                     (12)
#define ECG_SEG_TRANCONV_WEIGHT_PACK_SIZE                             (8)
#define ECG_SEG_DECODER_WEIGHT_PACK_SIZE                              (4)

#define ECG_SEG_ENCODER_CONVRELU_0_K_L                                (9)
#define ECG_SEG_ENCODER_CONVRELU0_0_K_C                               (4)
#define ECG_SEG_ENCODER_CONVRELU0_1_K_C                               (16)
#define ECG_SEG_ENCODER_CONVRELU1_0_K_C                               (32)
#define ECG_SEG_ENCODER_CONVRELU1_1_K_C                               (64)
#define ECG_SEG_ENCODER_CONVRELU2_0_K_C                               (128)
#define ECG_SEG_ENCODER_CONVRELU2_1_K_C                               (256)
#define ECG_SEG_ENCODER_CONVRELU3_0_K_C                               (512)
#define ECG_SEG_ENCODER_CONVRELU3_1_K_C                               (1024)
#define ECG_SEG_ENCODER_CONVRELU4_0_K_C                               (2048)
#define ECG_SEG_ENCODER_CONVRELU4_1_K_C                               (4096)
#define ECG_SEG_ENCODER_CONVRELU_0_K_DUMMING_PADDING                  (4)
#define ECG_SEG_ENCODER_CONVRELU_0_K_DUMMING_STRIDE                   (1)

#define ECG_SEG_TRANCONV_0_K_L                                        (8)
#define ECG_SEG_TRANCONV_0_K_C                                        (ECG_SEG_ENCODER_CONVRELU4_1_K_C)
#define ECG_SEG_TRANCONV_1_K_L                                        (8)
#define ECG_SEG_TRANCONV_1_K_C                                        (ECG_SEG_ENCODER_CONVRELU3_1_K_C)
#define ECG_SEG_TRANCONV_2_K_L                                        (8)
#define ECG_SEG_TRANCONV_2_K_C                                        (ECG_SEG_ENCODER_CONVRELU2_1_K_C)
#define ECG_SEG_TRANCONV_3_K_L                                        (8)
#define ECG_SEG_TRANCONV_3_K_C                                        (ECG_SEG_ENCODER_CONVRELU1_1_K_C)
#define ECG_SEG_TRANCONV_0_K_DUMMING_PADDING                          (3)
#define ECG_SEG_TRANCONV_0_K_DUMMING_STRIDE                           (2)

#define ECG_SEG_DECODER_CONVRELU_0_K_L                                (3)
#define ECG_SEG_DECODER_CONVRELU0_0_K_C                               (3072)
#define ECG_SEG_DECODER_CONVRELU0_1_K_C                               (1024)
#define ECG_SEG_DECODER_CONVRELU1_0_K_C                               (768)
#define ECG_SEG_DECODER_CONVRELU1_1_K_C                               (256)
#define ECG_SEG_DECODER_CONVRELU2_0_K_C                               (192)
#define ECG_SEG_DECODER_CONVRELU2_1_K_C                               (64)
#define ECG_SEG_DECODER_CONVRELU_0_K_DUMMING_PADDING                  (1)
#define ECG_SEG_DECODER_CONVRELU_0_K_DUMMING_STRIDE                   (1)

#define ECG_TEST_SIGNAL_ORI_C                                         (1)
#define ECG_TEST_SIGNAL_ORI_L                                         (2000)
#define ECG_TEST_SIGNAL_MID1_ORI_L                                    (1000)
#define ECG_TEST_SIGNAL_MID2_ORI_L                                    (500)
#define ECG_TEST_SIGNAL_MID3_ORI_L                                    (250)
#define ECG_TEST_SIGNAL_MID4_ORI_L                                    (125)
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
#define ECG_SINGAL_MID1_INPUT_IND                                     (0)
#define ECG_SIGNAL_MID1_0_OUTPUT_C                                    (8)
#define ECG_SIGNAL_MID1_0_OUTPUT_IND                                  (16)
#define ECG_SIGNAL_MID1_1_OUTPUT_C                                    (8)
#define ECG_SIGNAL_MID1_1_OUTPUT_IND                                  (8)
#define ECG_SIGNAL_TRAN_MID1_OUTPUT_C                                 (16)
#define ECG_SIGNAL_TRAN_MID1_OUTPUT_IND                               (16)
#define ECG_SIGNAL_DECODER_MID1_INPUT_C                               (24)
#define ECG_SIGNAL_DECODER_MID1_INPUT_IND                             (8)
#define ECG_SIGNAL_DECODER_MID1_0_OUTPUT_C                            (8)
#define ECG_SIGNAL_DECODER_MID1_0_OUTPUT_IND                          (0)
#define ECG_SIGNAL_DECODER_MID1_1_OUTPUT_C                            (8)
#define ECG_SIGNAL_DECODER_MID1_1_OUTPUT_IND                          (8)

#define ECG_SIGNAL_MID2_INPUT_C                                       (8)
#define ECG_SIGNAL_MID2_0_ORI_C                                       (16)
#define ECG_SIGNAL_MID2_ORI_L                                         (ECG_TEST_SIGNAL_MID2_ORI_L)
#define ECG_SIGNAL_MID2_K_L                                           (ECG_TEST_SIGNAL_K_L)
#define ECG_SIGNAL_MID2_PADDING                                       (ECG_TEST_SIGNAL_PADDING)
#define ECG_SIGNAL_MID2_STRIDE                                        (ECG_TEST_SIGNAL_STRIDE)
#define ECG_SIGNAL_MID2_MAX_C                                         (64)
#define ECG_SIGNAL_MID2_INPUT_IND                                     (0)
#define ECG_SIGNAL_MID2_0_OUTPUT_C                                    (16)
#define ECG_SIGNAL_MID2_0_OUTPUT_IND                                  (32)
#define ECG_SIGNAL_MID2_1_OUTPUT_C                                    (16)
#define ECG_SIGNAL_MID2_1_OUTPUT_IND                                  (16)
#define ECG_SIGNAL_TRAN_MID2_OUTPUT_C                                 (32)
#define ECG_SIGNAL_TRAN_MID2_OUTPUT_IND                               (32)
#define ECG_SIGNAL_DECODER_MID2_INPUT_C                               (48)
#define ECG_SIGNAL_DECODER_MID2_INPUT_IND                             (16)
#define ECG_SIGNAL_DECODER_MID2_0_OUTPUT_C                            (16)
#define ECG_SIGNAL_DECODER_MID2_0_OUTPUT_IND                          (0)
#define ECG_SIGNAL_DECODER_MID2_1_OUTPUT_C                            (16)
#define ECG_SIGNAL_DECODER_MID2_1_OUTPUT_IND                          (16)

#define ECG_SIGNAL_MID3_INPUT_C                                       (16)
#define ECG_SIGNAL_MID3_0_ORI_C                                       (32)
#define ECG_SIGNAL_MID3_ORI_L                                         (ECG_TEST_SIGNAL_MID3_ORI_L)
#define ECG_SIGNAL_MID3_K_L                                           (ECG_TEST_SIGNAL_K_L)
#define ECG_SIGNAL_MID3_PADDING                                       (ECG_TEST_SIGNAL_PADDING)
#define ECG_SIGNAL_MID3_STRIDE                                        (ECG_TEST_SIGNAL_STRIDE)
#define ECG_SIGNAL_MID3_MAX_C                                         (128)
#define ECG_SIGNAL_MID3_INPUT_IND                                     (0)
#define ECG_SIGNAL_MID3_0_OUTPUT_C                                    (32)
#define ECG_SIGNAL_MID3_0_OUTPUT_IND                                  (64)
#define ECG_SIGNAL_MID3_1_OUTPUT_C                                    (32)
#define ECG_SIGNAL_MID3_1_OUTPUT_IND                                  (32)
#define ECG_SIGNAL_TRAN_MID3_OUTPUT_C                                 (64)
#define ECG_SIGNAL_TRAN_MID3_OUTPUT_IND                               (64)
#define ECG_SIGNAL_DECODER_MID3_INPUT_C                               (96)
#define ECG_SIGNAL_DECODER_MID3_INPUT_IND                             (32)
#define ECG_SIGNAL_DECODER_MID3_0_OUTPUT_C                            (32)
#define ECG_SIGNAL_DECODER_MID3_0_OUTPUT_IND                          (0)
#define ECG_SIGNAL_DECODER_MID3_1_OUTPUT_C                            (32)
#define ECG_SIGNAL_DECODER_MID3_1_OUTPUT_IND                          (32)

#define ECG_SIGNAL_MID4_INPUT_C                                       (32)
#define ECG_SIGNAL_MID4_0_ORI_C                                       (64)
#define ECG_SIGNAL_MID4_ORI_L                                         (ECG_TEST_SIGNAL_MID4_ORI_L)
#define ECG_SIGNAL_MID4_K_L                                           (ECG_TEST_SIGNAL_K_L)
#define ECG_SIGNAL_MID4_PADDING                                       (ECG_TEST_SIGNAL_PADDING)
#define ECG_SIGNAL_MID4_STRIDE                                        (ECG_TEST_SIGNAL_STRIDE)
#define ECG_SIGNAL_MID4_MAX_C                                         (128)
#define ECG_SIGNAL_MID4_INPUT_IND                                     (0)
#define ECG_SIGNAL_MID4_0_OUTPUT_C                                    (64)
#define ECG_SIGNAL_MID4_0_OUTPUT_IND                                  (64)
#define ECG_SIGNAL_MID4_1_OUTPUT_C                                    (64)
#define ECG_SIGNAL_MID4_1_OUTPUT_IND                                  (0)

#define ECG_SIG2COL_MAX_OUT_L                                         (ECG_TEST_SIGNAL_MAX_OUT_L)
#define ECG_SIG2COL_MAX_K_L                                           (ECG_TEST_SIGNAL_MAX_K_L)

#define ECG_MIDDLE_FEATURE_GROUP_NUM                                  (5)
#define ECG_CONV_RELU_FUSE_GROUP_NUM                                  (19)
#define ECG_TRANCONV_GROUP_NUM                                        (4)
#define ECG_OUTPUT_PRED_GROUP_NUM                                     (1)

#endif
