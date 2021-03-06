/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#ifndef CONVOLUTIONSEPARABLE_COMMON_H
#define CONVOLUTIONSEPARABLE_COMMON_H



#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)



////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolution2dRowCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

extern "C" void convolution2dColumnCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

extern "C" void convolution3dRowCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int imageD,
    int kernelR
);

extern "C" void convolution3dColumnCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int imageD,
    int kernelR
);

extern "C" void convolution3dDepthCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int imageD,
    int kernelR
);


////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////

// 2d
extern "C" void setConvolutionKernel(float *h_Kernel);

extern "C" void convolution2dRowsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH
);

extern "C" void convolution2dColumnsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH
);

// 3d

extern "C" void convolution3dRowsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD
);

extern "C" void convolution3dColumnsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD
);

extern "C" void convolution3dDepthGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD
);

#endif
