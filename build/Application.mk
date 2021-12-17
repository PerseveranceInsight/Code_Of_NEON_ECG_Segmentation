APP_PROJECT_PATH := $(call my-dir)
APP_ABI := arm64-v8a

ifeq ($(EN_DEBUG),true)
APP_OPTIM := debug
APP_DEBUG := true
APP_STL := c++_shared # Or system, or none.
APP_CFLAGS := -fsanitize=address -fno-omit-frame-pointer -fsanitize=undefined -fsanitize=leak
# APP_CFLAGS += -fsanitize=shadow-call-stack -fsanitize=cfi
# APP_CFLAGS += -fsanitize=unreachable -fsanitize=signed-integer-overflow
# APP_CFLAGS += -fsanitize=shift -fsanitize=return
# APP_CFLAGS += -fsanitize=pointer-overflow -fsanitize=integer-divide-by-zero
# APP_CFLAGS += -fsanitize=implicit-integer-sign-change
# APP_CFLAGS += -fsanitize=implicit-unsigned-integer-truncation
# APP_CFLAGS += -fsanitize=implicit-signed-integer-truncation
# APP_CFLAGS += -fsanitize=float-divide-by-zero
# APP_CFLAGS += -fsanitize=float-cast-overflow
APP_LDFLAGS := -fsanitize=address -fsanitize=undefined
else
APP_OPTIM := release
APP_DEBUG := false
APP_STL := c++_shared # Or system, or none.
endif

ifeq ($(EN_DEBUG_SYM),true)
APP_STRIP_MODE := none
endif
