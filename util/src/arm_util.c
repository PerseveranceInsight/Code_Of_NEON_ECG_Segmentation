#include <stdio.h>
#include <time.h>
#include <arm_neon.h>
#include "arm_util.h"

double now_ns(void)
{
    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    return 1000.0*res.tv_sec + (double)res.tv_nsec/1e6;
}

void ree_dbg_neon_u32x4_t(uint32x4_t vec_dbg)
{
    uint32_t dbg_buf[4] = {0};
    vst1q_u32(dbg_buf, vec_dbg);
    ree_log(LOG_DEBUG, "%04d %04d %04d %04d", dbg_buf[0], dbg_buf[1], dbg_buf[2], dbg_buf[3]);
}

void ree_dbg_neon_u32x4x4_t(uint32x4x4_t vec_dbg)
{
    uint32_t dbg_buf[16] = {0};
    vst4q_u32(dbg_buf, vec_dbg);
    ree_log(LOG_DEBUG, "%04d %04d %04d %04d", dbg_buf[0], dbg_buf[1], dbg_buf[2], dbg_buf[3]);
    ree_log(LOG_DEBUG, "%04d %04d %04d %04d", dbg_buf[4], dbg_buf[5], dbg_buf[6], dbg_buf[7]);
    ree_log(LOG_DEBUG, "%04d %04d %04d %04d", dbg_buf[8], dbg_buf[9], dbg_buf[10], dbg_buf[11]);
    ree_log(LOG_DEBUG, "%04d %04d %04d %04d", dbg_buf[12], dbg_buf[13], dbg_buf[14], dbg_buf[15]);
}

void ree_dbg_neon_fp32x4x4_t(float32x4x4_t vec_dbg)
{
    float dbg_buf[16] = {0.0};
    vst4q_f32(dbg_buf, vec_dbg);
    ree_log(LOG_DEBUG, "%1.1f %1.1f %1.1f %1.1f", dbg_buf[0], dbg_buf[1], dbg_buf[2], dbg_buf[3]);
    ree_log(LOG_DEBUG, "%1.1f %1.1f %1.1f %1.1f", dbg_buf[4], dbg_buf[5], dbg_buf[6], dbg_buf[7]);
    ree_log(LOG_DEBUG, "%1.1f %1.1f %1.1f %1.1f", dbg_buf[8], dbg_buf[9], dbg_buf[10], dbg_buf[11]);
    ree_log(LOG_DEBUG, "%1.1f %1.1f %1.1f %1.1f", dbg_buf[12], dbg_buf[13], dbg_buf[14], dbg_buf[15]);
}
