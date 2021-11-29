#include <android/log.h>
#include <arm_neon.h>
#ifndef __ARM_UTIL_H__
#define __ARM_UTIL_H__

typedef enum LOG_LEVEL {
    LOG_VERBOSE = 2,
    LOG_DEBUG = 3,
    LOG_INFO = 4,
    LOG_WARN = 5,
    LOG_ERROR = 6,
} LOG_LEVEL_T;

#define DEFAULT_LOG_LEVEL (LOG_DEBUG)

#ifdef __DEBUG__

#define ree_log(LOG_LEVEL, x...)                                                        \
{                                                                                       \
    if (LOG_LEVEL >= DEFAULT_LOG_LEVEL) {                                               \
        switch (LOG_LEVEL) {                                                            \
            case LOG_VERBOSE:                                                           \
                __android_log_print(ANDROID_LOG_WARN, "ECG_SEG", x);                    \
                break;                                                                  \
            case LOG_DEBUG:                                                             \
                __android_log_print(ANDROID_LOG_DEBUG, "ECG_SEG", x);                   \
                break;                                                                  \
            case LOG_INFO:                                                              \
                __android_log_print(ANDROID_LOG_INFO, "ECG_SEG", x);                    \
                break;                                                                  \
            case LOG_WARN:                                                              \
                __android_log_print(ANDROID_LOG_WARN, "ECG_SEG", x);                    \
                break;                                                                  \
            case LOG_ERROR:                                                             \
                __android_log_print(ANDROID_LOG_ERROR, "ECG_SEG", x);                   \
                break;                                                                  \
        }                                                                               \
        printf("%s [%d] ", __FILE__, __LINE__);                                         \
        printf(x);                                                                      \
        printf("\n");                                                                   \
    }                                                                                   \
}

#define FUNC_ENTRANCE_LOG       ree_log(LOG_DEBUG, "%s enters", __func__);
#define FUNC_EXIT_LOG           ree_log(LOG_DEBUG, "%s leaves", __func__);
#define ree_printf(LOG_LEVEL, x...)                                                     \
{                                                                                       \
    if (LOG_LEVEL >= DEFAULT_LOG_LEVEL) {                                               \
        printf(x);                                                                      \
    }                                                                                   \
}
#else
#define ree_log(LOG_LEVEL, x...)                                           do {} while (0)
#define FUNC_ENTANGE_LOG                                                   do {} while (0)
#define FUNC_EXIT_LOG                                                      do {} while (0)
#define ree_printf(LOG_LEVEL, x...)                                        do {} while (0)
#endif

#ifndef NULL
#define NULL                                                                    ((void*)0)
#endif
#define ree_malloc(size)                                                      malloc(size)
#define ree_set(dst, value, size)                                 memset(dst, value, size)
#define ree_cpy(dst, src, size)                                     memcpy(dst, src, size)
#define ree_free(src)                                                                    \
{                                                                                        \
    if (src)                                                                             \
    {                                                                                    \
        free(src);                                                                       \
        src = NULL;                                                                      \
    }                                                                                    \
}

#define ree_fopen(file_name, mode)                         fopen(file_name, mode)   
#define ree_fclose(src_file)                                                        \
{                                                                                   \
    if (src_file)                                                                   \
    {                                                                               \
        fclose(src_file);                                                           \
        src_file = NULL;                                                            \
    }                                                                               \
}

#define ree_check_fopen(file_ptr, file_name, mode, EXIT_TAG)                        \
{                                                                                   \
    file_ptr = ree_fopen(file_name, mode);                                          \
    if (!file_ptr)                                                                  \
    {                                                                               \
        ree_log(LOG_ERROR, "%s can't open %s", __func__, file_name);                \
        goto EXIT_TAG;                                                              \
    }                                                                               \
}

#define ree_file_size(file_size, file_ptr)                                          \
{                                                                                   \
    fseek(file_ptr, 0L, SEEK_END);                                                  \
    file_size = (size_t)ftell(file_ptr);                                            \
    ree_log(LOG_DEBUG, "%s FILE's size is %zu", __func__, file_size);               \
    ree_fclose(file_ptr);                                                           \
}

#define ree_file_read(file_ptr, p_buffer, file_size, read_size)                     \
{                                                                                   \
    read_size = fread(p_buffer, sizeof(char), file_size, file_ptr);                 \
    ree_fclose(file_ptr);                                                           \
    ree_log(LOG_DEBUG, "%s read_size is %zu", __func__, read_size);                 \
}

#define ree_file_write(file_ptr, p_buffer, file_name, file_size)                    \
{                                                                                   \
    if (file_ptr && p_buffer)                                                       \
    {                                                                               \
        ree_log(LOG_DEBUG, "%s saves %s", __func__, file_name);                     \
        fwrite(p_buffer, sizeof(uint8_t), file_size, file_ptr);                     \
        ree_fclose(file_ptr);                                                       \
    } else                                                                          \
    {                                                                               \
        ree_log(LOG_ERROR, "%s prepares to save %s failed", __func__, file_name);   \
    }                                                                               \
}

#define ree_check_null_exit(ptr, EXIT_LABEL, ERROR_LOG, ...)                        \
{                                                                                   \
    if (!ptr) {                                                                     \
        ree_log(LOG_ERROR, ERROR_LOG, ##__VA_ARGS__);                               \
        goto EXIT_LABEL;                                                            \
    }                                                                               \
}

#define ree_check_null_exit_retval(ptr, retval, RET_CODE, EXIT_LABEL, ERROR_LOG, ...) \
{                                                                                     \
    if (!ptr) {                                                                       \
        retval = RET_CODE;                                                            \
        ree_log(LOG_ERROR, ERROR_LOG, ##__VA_ARGS__);                                 \
        goto EXIT_LABEL;                                                              \
    }                                                                                 \
}

#define ree_check_true_exit(check_v, EXIT_LABEL, ERROR_LOG, ...)                      \
{                                                                                     \
    if (check_v) {                                                                    \
        ree_log(LOG_ERROR, ERROR_LOG, ##__VA_ARGS__);                                 \
        goto EXIT_LABEL;                                                              \
    }                                                                                 \
}

#define ree_check_true_exit_retval(check_v, retv, RET_CODE, EXIT_LABEL, ERROR_LOG, ...) \
{                                                                                       \
    if (check_v) {                                                                      \
        retval = RET_CODE;                                                              \
        ree_log(LOG_ERROR, ERROR_LOG, ##__VA_ARGS__);                                   \
        goto EXIT_LABEL;                                                                \
    }                                                                                   \
}

#define REE_UNUSED __attribute__((unused))

double now_ns(void);
void ree_dbg_neon_u32x4_t(uint32x4_t vec_dbg);
void ree_dbg_neon_hex_u32x4_t(uint32x4_t vec_dbg);
void ree_dbg_neon_u32x4x4_t(uint32x4x4_t vec_dbg);
void ree_dbg_neon_fp32x2_t(float32x2_t vec_dbg);
void ree_dbg_neon_fp32x4_t(float32x4_t vec_dbg);
void ree_dbg_neon_fp32x4x4_t(float32x4x4_t vec_dbg);
#endif
