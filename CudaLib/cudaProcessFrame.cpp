/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 /* This example demonstrates how to use the Video Decode Library with CUDA 
  * bindings to interop between CUDA and DX9 textures for the purpose of post
  * processing video.
  */

#include "cudaProcessFrame.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda.h>
#include <builtin_types.h>
#include <cutil_inline.h>

// These store the matrix for YUV2RGB transformation
__constant__ float  constHueColorSpaceMat[9];
__constant__ float  constAlpha;


extern "C"
CUresult  updateConstantMemory_drvapi(CUmodule module, float *hueCSC)
{
    CUdeviceptr  d_constHueCSC, d_constAlpha;
    unsigned int d_cscBytes, d_alphaBytes;

    // First grab the global device pointers from the CUBIN
    cuModuleGetGlobal(&d_constHueCSC,  &d_cscBytes  , module, "constHueColorSpaceMat");
    cuModuleGetGlobal(&d_constAlpha ,  &d_alphaBytes, module, "constAlpha"           );

    CUresult error = CUDA_SUCCESS;

    // Copy the constants to video memory
    cuMemcpyHtoD( d_constHueCSC,            
                  reinterpret_cast<const void *>(hueCSC),
                  d_cscBytes);
	cutilDrvCheckMsg("cuMemcpyHtoD (d_constHueCSC) copy to Constant Memory failed");
        

    uint32 cudaAlpha      = ((uint32)0xff<< 24);

    cuMemcpyHtoD( constAlpha,
                  reinterpret_cast<const void *>(&cudaAlpha),                         
                  d_alphaBytes);
	cutilDrvCheckMsg("cuMemcpyHtoD (constAlpha) copy to Constant Memory failed");

    return error;
}

extern "C"
void setColorSpaceMatrix(eColorSpace CSC, float *hueCSC, float hue)
{
    float hueSin = sin(hue);
    float hueCos = cos(hue);

    if (CSC == ITU601) {
        //CCIR 601
	    hueCSC[0] = 1.1644f;
	    hueCSC[1] = hueSin * 1.5960f;
	    hueCSC[2] = hueCos * 1.5960f;
	    hueCSC[3] = 1.1644f;
	    hueCSC[4] = (hueCos * -0.3918f) - (hueSin * 0.8130f);
	    hueCSC[5] = (hueSin *  0.3918f) - (hueCos * 0.8130f);  
	    hueCSC[6] = 1.1644f;
	    hueCSC[7] = hueCos *  2.0172f;
	    hueCSC[8] = hueSin * -2.0172f;
    } else if (CSC == ITU709) {
        //CCIR 709
        hueCSC[0] = 1.0f;
        hueCSC[1] = hueSin * 1.57480f;
        hueCSC[2] = hueCos * 1.57480f;
        hueCSC[3] = 1.0;
        hueCSC[4] = (hueCos * -0.18732f) - (hueSin * 0.46812f);
        hueCSC[5] = (hueSin *  0.18732f) - (hueCos * 0.46812f);  
        hueCSC[6] = 1.0f;
        hueCSC[7] = hueCos *  1.85560f;
        hueCSC[8] = hueSin * -1.85560f;    
    }
}

// We call this function to launch the CUDA kernel (NV12 to ARGB).  
extern "C"
CUresult  cudaLaunchNV12toARGBDrv(  CUdeviceptr d_srcNV12, uint32 nSourcePitch,   
								    CUdeviceptr d_dstARGB, uint32 nDestPitch,
                                    uint32 width,          uint32 height,
                                    CUfunction fpFunc, CUstream streamID)
{
    // EAch kernel thread will output 2 pixels at a time.  The grid size width is half
    // as large because of this
    dim3 block(32,16);
    dim3 grid((width+(2*block.x-1))/(2*block.x), (height+(block.y-1))/block.y); 

    // setup kernel execution parameters (block dimensions)
    cutilDrvSafeCall (cuFuncSetBlockShape( fpFunc, block.x, block.y, 1 ));
    int offset = 0;

    // This method calls cuParamSetv() to pass device pointers also allows the ability to pass 64-bit device pointers
	// This method calls cuParamSetv() to pass device pointers also allows the ability to pass 64-bit device pointers

	// device pointer for Source Surface
	cutilDrvSafeCall (cuParamSetv   ( fpFunc, offset, &d_srcNV12,  sizeof(d_srcNV12) ));   
	offset += sizeof(d_srcNV12);

	// set the Source pitch
    ALIGN_OFFSET(offset, __alignof(nSourcePitch));
    cutilDrvSafeCall (cuParamSeti   ( fpFunc, offset, nSourcePitch ));
	offset += sizeof(nSourcePitch);

	// device pointer for Destination Surface
    cutilDrvSafeCall (cuParamSetv   ( fpFunc, offset, &d_dstARGB,  sizeof(d_dstARGB) ));   
	offset += sizeof(d_dstARGB);

	//	set the Destination Pitch
    ALIGN_OFFSET(offset, __alignof(nDestPitch)); 
    cutilDrvSafeCall (cuParamSeti   ( fpFunc, offset, nDestPitch ));                   offset += sizeof(nDestPitch);

	// set the width of the image
	ALIGN_OFFSET(offset, __alignof(width));
    cutilDrvSafeCall (cuParamSeti   ( fpFunc, offset, width ));                        offset += sizeof(width);

	// set the height of the image
    ALIGN_OFFSET(offset, __alignof(height)); 
    cutilDrvSafeCall (cuParamSeti   ( fpFunc, offset, height ));                       offset += sizeof(height);

    cutilDrvSafeCall (cuParamSetSize( fpFunc, offset));

    // Launching the kernel, we need to pass in the grid dimensions
    CUresult error = cuLaunchGridAsync( fpFunc, grid.x, grid.y, streamID );

    return error;
}

