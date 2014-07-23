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



#include <assert.h>
#include <helper_cuda.h>
#include "convolutionSeparable_common.h"



////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[KERNEL_LENGTH];

extern "C" void setConvolutionKernel(float *h_Kernel)
{
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));
}


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////

// how many threads per block in x (total num threads: x*y)
#define   ROWS_BLOCKDIM_X 16

// how many threads per block in y
#define   ROWS_BLOCKDIM_Y 4

// how many pixels in x are convolved by each thread
#define ROWS_RESULT_STEPS 8

// these are the border pixels (loaded to support the kernel width for processing)
// the effective border width is ROWS_HALO_STEPS * ROWS_BLOCKDIM_X, which has to be
// larger or equal to the kernel radius to work
#define   ROWS_HALO_STEPS 1

__global__ void convolution2dRowsKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int pitch
)
{
    // shared memory of pixels for all threads (n=ROWS_BLOCKDIM_X * ROWS_BLOCKDIM_Y) of one block
    // note: up to Compute capability 3.5 this is 48kb
    //
    // in y: the number of threads (ROWS_BLOCKDIM_Y)
    // in x: the number of threads (ROWS_BLOCKDIM_X) * (ROWS_RESULT_STEPS=Number of processed pixels + twice the ROW_HALO_STEPS (whatever the fuck this is) )
    __shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    // Offset (in the input and output array!) to the left halo edge relative to the current block and thread that is processed, 
    // i.e. which pixels are we copying in this thread?
    //
    // blockIdx.x and blockIdx.y give us an index for the current block, so the pixel coordinates in x and y result from the number of threads per block,
    // or many also be called the blocksize ROWS_BLOCKDIM_X and ROWS_BLOCKDIM_Y
    //
    // threadIdx.x and threadIdx.y give use the corresponding index of the current thread within the block, so 
    //
    // This is simple in y, the line we are in is just the current block times the blocksize (#of threads in y) plus the current thread
    // It is a little more complicated in x. The threads are set one after another left of the actual data that is processed, and are
    // increased in ROWS_BLOCKDIM_X steps to fill up the shared memory. The amount of space next to the data that is convolved is determined
    // by ROWS_HALO_STEPS * ROWS_BLOCKDIM_X, so in the default version it is 16 pixels, which is the maximally supported kernel radius.
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    // set the input and output arrays to the right offset (actually the output is not at the right offset, but this is corrected later)
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    // Load main data
    // Start copying after the ROWS_HALO_STEPS, only the original data that will be convolved
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

    // Load left halo
    // If the data fetched is outside of the image (note: baseX can be <0 for the first block) , use a zero-out of bounds strategy
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        float sum = 0;

#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
        }

        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}

extern "C" void convolution2dRowsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH
)
{
    assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
    assert(imageH % ROWS_BLOCKDIM_Y == 0);

    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

    convolution2dRowsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 1

__global__ void convolution2dColumnsKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int pitch
)
{
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        float sum = 0;
#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}

extern "C" void convolution2dColumnsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH
)
{
    assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % COLUMNS_BLOCKDIM_X == 0);
    assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

    convolution2dColumnsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW
    );
    getLastCudaError("convolutionColumnsKernel() execution failed\n");
}

//
// 3d implementation
//

__global__ void convolution3dRowsKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD,
    int pitchX,
    int pitchY
)
{
    __shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = blockIdx.z;

    // set the input and output arrays to the right offset (actually the output is not at the right offset, but this is corrected later)
    d_Src += baseZ * pitchY * pitchX + baseY * pitchX + baseX;
    d_Dst += baseZ * pitchY * pitchX + baseY * pitchX + baseX;

    // Load main data
    // Start copying after the ROWS_HALO_STEPS, only the original data that will be convolved
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

    // Load left halo
    // If the data fetched is outside of the image (note: baseX can be <0 for the first block) , use a zero-out of bounds strategy
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        float sum = 0;

#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
        }

        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}

extern "C" void convolution3dRowsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD
)
{
    assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
    assert(imageH % ROWS_BLOCKDIM_Y == 0);

    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y, imageD);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

    convolution3dRowsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageD,
        imageW,
        imageH
    );
    getLastCudaError("convolution3dRowsKernel() execution failed\n");
}

__global__ void convolution3dColumnsKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD,
    int pitchX,
    int pitchY
)
{
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    const int baseZ = blockIdx.z + threadIdx.z;

    d_Src += baseZ * pitchY * pitchX + baseY * pitchX + baseX;
    d_Dst += baseZ * pitchY * pitchX + baseY * pitchX + baseX;

    //Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitchX];
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitchX] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitchX] : 0;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        float sum = 0;
#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitchX] = sum;
    }
}

extern "C" void convolution3dColumnsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD
)
{
    assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % COLUMNS_BLOCKDIM_X == 0);
    assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y), imageD);
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1);

    convolution3dColumnsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageD,
        imageW,
        imageH
    );
    getLastCudaError("convolution3dColumnsKernel() execution failed\n");
}

#define   DEPTH_BLOCKDIM_X 16
#define   DEPTH_BLOCKDIM_Z 8
#define DEPTH_RESULT_STEPS 8
#define   DEPTH_HALO_STEPS 1

__global__ void convolution3dDepthKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD,
    int pitchX,
    int pitchY
)
{
	// here it is [x][z], we leave out y as it has a size of 1
    __shared__ float s_Data[DEPTH_BLOCKDIM_X][(DEPTH_RESULT_STEPS + 2 * DEPTH_HALO_STEPS) * DEPTH_BLOCKDIM_Z + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * DEPTH_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y + threadIdx.y;
    const int baseZ = (blockIdx.z * DEPTH_RESULT_STEPS - DEPTH_HALO_STEPS) * DEPTH_BLOCKDIM_Z + threadIdx.z;

    d_Src += baseZ * pitchY * pitchX + baseY * pitchX + baseX;
    d_Dst += baseZ * pitchY * pitchX + baseY * pitchX + baseX;

    //Main data
#pragma unroll

    for (int i = DEPTH_HALO_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z] = d_Src[i * DEPTH_BLOCKDIM_Z * pitchX * pitchY];
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < DEPTH_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z] = (baseZ >= -i * DEPTH_BLOCKDIM_Z) ? d_Src[i * DEPTH_BLOCKDIM_Z * pitchX * pitchY] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS + DEPTH_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * DEPTH_BLOCKDIM_Z]= (imageD - baseZ > i * DEPTH_BLOCKDIM_Z) ? d_Src[i * DEPTH_BLOCKDIM_Z * pitchX * pitchY] : 0;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = DEPTH_HALO_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i++)
    {
        float sum = 0;
#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z + j];
        }

        d_Dst[i * DEPTH_BLOCKDIM_Z * pitchX * pitchY] = sum;
    }
}

extern "C" void convolution3dDepthGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int imageD
)
{
    assert(DEPTH_BLOCKDIM_Z * DEPTH_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % DEPTH_BLOCKDIM_X == 0);
    assert(imageD % (DEPTH_RESULT_STEPS * DEPTH_BLOCKDIM_Z) == 0);

    dim3 blocks(imageW / DEPTH_BLOCKDIM_X, imageH, imageD/ (DEPTH_RESULT_STEPS * DEPTH_BLOCKDIM_Z) );
    dim3 threads(DEPTH_BLOCKDIM_X, 1, DEPTH_BLOCKDIM_Z);

    convolution3dDepthKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageD,
        imageW,
        imageH
    );
    getLastCudaError("convolution3dDepthKernel() execution failed\n");
}

