LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := ecg_seg

LOCAL_CFLAGS += -D__DEBUG__ -Ofast
LOCAL_CFLAGS += -Wall -Werror 
LOCAL_CFLAGS += -Wno-error=unused-function -Wno-error=unused-variable -Wno-error=unused-label -Wno-error=return-type
LOCAL_CFLAGS += -Wno-error=sign-compare -Wno-error=multichar -Wno-error=implicit-fallthrough -Wno-error=vla
LOCAL_CFLAGS += -Wno-error=strict-prototypes
ifeq ($(EN_DEBUG_SYM), true)
LOCAL_CFLAGS += -g -ggdb
endif

ifeq ($(EN_MATRIX_DBG),true)
LOCAL_CFLAGS += -DMATRIX_DBG
endif

ifeq ($(EN_MODEL_DBG),true)
LOCAL_CFLAGS += -DMODEL_DBG
endif

ifeq ($(EN_SIGNAL_DBG),true)
LOCAL_CFLAGS += -DSIGNAL_DBG
endif

ifeq ($(EN_PACK_DEBUG),true)
LOCAL_CFLAGS += -DEN_PACK_DEBUG
endif

ifeq ($(EN_GEMM_DBG),true)
LOCAL_CFLAGS += -DEN_GEMM_DBG
endif

ifeq ($(EN_SIGNAL_DBG),true)
LOCAL_CFLAGS += -DEN_SIGNAL_DBG
endif

ifeq ($(EN_SIG2COL_DBG),true)
LOCAL_CFLAGS += -DEN_SIG2COL_DBG
endif

LOCAL_ARM_MODE := arm

PROJECT_SRC = $(LOCAL_PATH)/../src
PROJECT_INC = $(LOCAL_PATH)/../inc
PROJECT_UTIL_INC = $(LOCAL_PATH)/../util/inc
PROJECT_UTIL_SRC = $(LOCAL_PATH)/../util/src

LOCAL_C_INCLUDES += $(PROJECT_INC) \
					$(PROJECT_UTIL_INC)

LOCAL_SRC_FILES +=  ${PROJECT_SRC}/ecg_seg_main.c \
					${PROJECT_SRC}/ecg_seg_gemm.c \
					${PROJECT_SRC}/ecg_seg_matrix.c \
					${PROJECT_SRC}/ecg_seg_model.c \
					${PROJECT_SRC}/ecg_seg_sig2col.c \
					${PROJECT_SRC}/ecg_seg_signal.c \
				    ${PROJECT_UTIL_SRC}/arm_util.c


LOCAL_LDLIBS := -lm -llog
LOCAL_LDFLAGS := -nodefaultlibs -lc -lm -ldl
LOCAL_LDLIBS :=  -llog -lm

include $(BUILD_EXECUTABLE)
