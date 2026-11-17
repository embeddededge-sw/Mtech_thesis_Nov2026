// SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * cnn_accel_uapi.h - Userspace API for CNN accelerator driver
 *
 * Shared between kernel driver and userspace application.
 * Uses __u32/__s32 types which resolve correctly in both contexts.
 */
#ifndef CNN_ACCEL_UAPI_H
#define CNN_ACCEL_UAPI_H

#ifdef __KERNEL__
#include <linux/ioctl.h>
#include <linux/types.h>
#else
#include <sys/ioctl.h>
#include <stdint.h>
typedef uint32_t __u32;
typedef int32_t __s32;
#endif

#define CNN_INPUT_SIZE      784
#define CNN_MAX_WEIGHTS     5738

struct cnn_weight_data {
    __u32 count;
    __u32 *data;
};

struct cnn_inference_req {
    __u32 input_size;
    __u32 image[CNN_INPUT_SIZE];
    __s32 result;
};

#define CNN_IOCTL_MAGIC                 'C'
#define CNN_IOCTL_LOAD_WEIGHTS          _IOW(CNN_IOCTL_MAGIC, 1, struct cnn_weight_data)
#define CNN_IOCTL_RUN_INFERENCE         _IOWR(CNN_IOCTL_MAGIC, 2, struct cnn_inference_req)

/* NEW IOCTL FOR NON-HLS RTL MODEL */
#define CNN_IOCTL_RUN_NONHLS             _IOWR(CNN_IOCTL_MAGIC, 3, struct cnn_inference_req)

#endif /* CNN_ACCEL_UAPI_H */
