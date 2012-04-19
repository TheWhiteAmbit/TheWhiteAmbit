//The White Ambit, Rendering-Framework
//Copyright (C) 2009  Moritz Strickhausen

//This program is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with this program.  If not, see <http://www.gnu.org/licenses/>.


#include <stdio.h>
#include <stdlib.h>


__global__ void kernelYuvKonvert(unsigned char* surfaceSrc, unsigned char* surfaceDst,  size_t width, size_t height, size_t pitchSrc, size_t pitchDst)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixelSrc = (float*)(surfaceSrc + y*pitchSrc) + 4*x;
	
	int offsetUV = height*pitchDst;
	unsigned char* pixelY = (unsigned char*)(surfaceDst + y * pitchDst) + (x/4);
	unsigned char* pixelUV = (unsigned char*)(surfaceDst + (y/2) * pitchDst) + (x/8) + offsetUV;

	pixelY[0]=(unsigned char) min(255.0f, pixelSrc[0]*255.0f);
	pixelUV[0]=(unsigned char) min(255.0f, pixelSrc[1]*255.0f);
	pixelUV[1]=(unsigned char) min(255.0f, pixelSrc[2]*255.0f);
}

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 8

extern "C" 
void cuda_yuv_transform(void* surfaceSrc, void* surfaceDst, size_t width, size_t height, size_t pitchSrc, size_t pitchDst)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3( BLOCKDIM_X, BLOCKDIM_Y ); // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3( (width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y );

	
	kernelYuvKonvert<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitchSrc, pitchDst );	

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("cuda_yuv_transform() failed to launch error = %d\n", error);
	}
}



