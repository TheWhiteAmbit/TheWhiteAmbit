/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.1415926536f

/* 
 * Paint a 2D surface with a moving bulls-eye pattern.  The "face" parameter selects
 * between 6 different colors to use.  We will use a different color on each face of a
 * cube map.
 */
__global__ void cuda_kernel_texture_cube(char* surface, int width, int height, size_t pitch, int face, float t)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned char* pixel;
	
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels	
	if (x >= width || y >= height) return;
	
	// get a pointer to this pixel
	pixel = (unsigned char*)(surface + y*pitch) + 4*x;
	
	// populate it
	float theta_x = (2.0f*x)/width  - 1.0f;
	float theta_y = (2.0f*y)/height - 1.0f;
	float theta = 2.0f*PI*sqrt(theta_x*theta_x + theta_y*theta_y);
	unsigned char value = 255*( 0.6f + 0.4f*cos(theta + t) );
	
	pixel[3] = 255; // alpha
	if (face%2)
	{
		pixel[0] =    // blue
		pixel[1] =    // green
		pixel[2] = 0; // red
		pixel[face/2] = value;
	}
	else
	{
		pixel[0] =        // blue
		pixel[1] =        // green
		pixel[2] = value; // red
		pixel[face/2] = 0;
	}
}

extern "C" 
void cuda_texture_cube(void* surface, int width, int height, size_t pitch, int face, float t)
{
    cudaError_t error = cudaSuccess;

    dim3 Db = dim3( 16, 16 ); // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3( (width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y );

    cuda_kernel_texture_cube<<<Dg,Db>>>( (char*)surface, width, height, pitch, face, t );

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("cuda_kernel_texture_cube() failed to launch error = %d\n", error);
    }
}

