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



#include "rendering.h"
#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
//#include <cstdlib> 
//#include <ctime> 
//#include <iostream>
#include "bsp_raytracer.cu"


__global__ void kernelClearTexture(unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* matrices, float4 blend)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;

	pixel[0]=blend.x;
	pixel[1]=blend.y;
	pixel[2]=blend.z;
	pixel[3]=blend.w;
}	

__global__ void kernelCopyTexture(unsigned char* surfaceSrc, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
	float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;

	pixel[0]=ray[0];
	pixel[1]=ray[1];
	pixel[2]=ray[2];
	pixel[3]=ray[3];
}


__global__ void kernelShadeTexture(unsigned char* surfaceSrc, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
	float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
	if( ray[3]==0.0f ){
		pixel[0]=ray[0];
		pixel[1]=ray[1];
		pixel[2]=ray[2];
		pixel[3]=0.0f;
		return;
	}

	//read the needed matrices for projection
	float4* projection;
	projection = (float4*)(matrices+4*2);
	float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);

	float4 light;
	light.x=0.5773f;
	light.y=0.5773f;
	light.z=-0.5773f;
	light.w=0.0f;

	float4 rayDir;
	float4 rayOrig;
	float4 raySrf;
	reflectRay(x, y, width, height, ray, &rayDir, &rayOrig, &raySrf, projection, invWorldView);

	//init shading color and bring lightsource position
	float l=dot(light,raySrf);

	//float4 halfVec;
	//halfVec.x=(light.x+ray[0])*.5f;
	//halfVec.y=(light.y+ray[1])*.5f;
	//halfVec.z=(light.z+ray[2])*.5f;
	//halfVec.w=0.0f;
	//halfVec=normalize3(halfVec);
	//float blinn=pow(dot(raySrf, halfVec), 64);
	float blinn=0.0f;
	//l=0.0f;

	pixel[0] = (l+GOOCHY)*LIGHT+blinn; // red
	pixel[1] = (l+GOOCHY)*LIGHT+blinn; // green
	pixel[2] = (l+GOOCHY)*LIGHT+blinn; // blue
	pixel[3] = ray[3];
}	

__global__ void kernelBacktrackTextureBlend(unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
	float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
	float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;
	
	float ray_x=(ray[0])*width;
	float ray_y=(float)height-(ray[1]*(float)height);
	ray_x=max(2.0f, min((float)width-2, ray_x));
	ray_y=max(2.0f, min((float)height-2, ray_y));

	float4 before = tex2DBilinear(surfaceSrc1, ray_x, ray_y, pitch);
	
	if(ray[3]==0.0){
		pixel[0]=ray[0]*(1.0f-rand.w);
		pixel[1]=ray[1]*(1.0f-rand.w);
		pixel[2]=ray[2]*(1.0f-rand.w);
		pixel[3]=ray[3]*(1.0f-rand.w);
		return;
	}

	pixel[0]=before.x;
	pixel[1]=before.y;
	pixel[2]=before.z;
	pixel[3]=before.w;
}

__global__ void kernelBacktrackShadowBlend(unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
	float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
	float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;
	
	float ray_x=(ray[0])*width;
	float ray_y=(float)height-(ray[1]*(float)height);
	ray_x=max(1.0f, min((float)width-2, ray_x));
	ray_y=max(1.0f, min((float)height-2, ray_y));

	//float4 before = tex2DBilinear(surfaceSrc1, ray_x, ray_y, pitch);
	float* before = tex2D(surfaceSrc1, ray_x, ray_y, pitch);

	pixel[0]=0.5f;
	pixel[1]=0.5f;
	pixel[2]=0.5f;

	//map the shadow
	if(ray[2] < before[3]){
		pixel[0]=0.0f;
		pixel[1]=0.0f;
		pixel[2]=0.0f;
	} else {
		pixel[0]=1.0f;
		pixel[1]=1.0f;
		pixel[2]=1.0f;
	}

	//float dim=(1.0f/before.w-1.0f/ray[2]);
	//pixel[0]=dim;
	//pixel[1]=dim;
	//pixel[2]=dim;
}

__global__ void kernelBacktrackShadowBlend2(unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
	float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
	
	float ray_x=(ray[0])*width;
	float ray_y=(float)height-(ray[1]*(float)height);
	ray_x=max(1.0f, min((float)width-2, ray_x ));
	ray_y=max(1.0f, min((float)height-2, ray_y ));

	float* before = tex2D(surfaceSrc1, ray_x + .5f, ray_y + .5f, pitch);

	//map the shadow
	if(ray[2] < before[3]){
		pixel[0]=ray[0]*.5;
		pixel[1]=ray[1]*.5;
		pixel[2]=ray[2]*.5;
	} else {
		pixel[0]=ray[0];
		pixel[1]=ray[1];
		pixel[2]=ray[2];
	}
}

__global__ void kernelPolygonMapBlend(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
	float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
	
	float ray_x=(ray[0])*width;
	float ray_y=(float)height-(ray[1]*(float)height);
	ray_x=max(1.0f, min((float)width-2, ray_x ));
	ray_y=max(1.0f, min((float)height-2, ray_y ));

	float* before = tex2D(surfaceSrc1, ray_x + .5f, ray_y + .5f, pitch);

	//map the shadow
	if(ray[2] < before[3]){
		pixel[0]=ray[0]*.5;
		pixel[1]=ray[1]*.5;
		pixel[2]=ray[2]*.5;
	} else {
		pixel[0]=ray[0];
		pixel[1]=ray[1];
		pixel[2]=ray[2];
	}
}

__global__ void kernelPolygonMapBlend2(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
	float* ray = (float*)(surfaceSrc1 + y*pitch) + 4*x;
	
	float ray_x=(ray[0])*width;
	float ray_y=(float)height-(ray[1]*(float)height);
	ray_x=max(1.0f, min((float)width-2, ray_x ));
	ray_y=max(1.0f, min((float)height-2, ray_y ));

	float* before = tex2D(surfaceSrc, ray_x + .5f, ray_y + .5f, pitch);

	//map the shadow
	if(ray[2] < before[3]){
		pixel[0]=ray[0]*.5;
		pixel[1]=ray[1]*.5;
		pixel[2]=ray[2]*.5;
	} else {
		pixel[0]=ray[0];
		pixel[1]=ray[1];
		pixel[2]=ray[2];
	}
}

__global__ void kernelPolygonMap3(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
	float* ray = (float*)(surfaceSrc1 + y*pitch) + 4*x;
	
	float ray_x=(ray[0])*width;
	float ray_y=(float)height-(ray[1]*(float)height);
	ray_x=max(1.0f, min((float)width-2, ray_x));
	ray_y=max(1.0f, min((float)height-2, ray_y));

	float* before = tex2D(surfaceSrc, ray_x + .5f, ray_y + .5f, pitch);

	//map the shadow
	if(ray[2] < before[3]){
		pixel[0]=before[0]*.5;
		pixel[1]=before[1]*.5;
		pixel[2]=before[2]*.5;
	} else {
		pixel[0]=before[0];
		pixel[1]=before[1];
		pixel[2]=before[2];
	}
}

__global__ void kernelPolygonMapBlend3(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
	float* ray = (float*)(surfaceSrc1 + y*pitch) + 4*x;
	
	float ray_x=(ray[0])*width;
	float ray_y=(float)height-(ray[1]*(float)height);
	ray_x=max(1.0f, min((float)width-2, ray_x + rand.x ));
	ray_y=max(1.0f, min((float)height-2, ray_y + rand.y ));

	float* before = tex2D(surfaceSrc, ray_x, ray_y, pitch);

	//map the shadow
	if(ray[2] < before[3]){
		pixel[0]=before[0]*.5;
		pixel[1]=before[1]*.5;
		pixel[2]=before[2]*.5;
	} else {
		pixel[0]=before[0];
		pixel[1]=before[1];
		pixel[2]=before[2];
	}
}

__global__ void kernelPolygonMapBlend4(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
	float* pixel1 = (float*)(surfaceDst1 + y*pitch) + 4*x;
	float* pixel2 = (float*)(surfaceDst2 + y*pitch) + 4*x;
	float* pixel3 = (float*)(surfaceDst3 + y*pitch) + 4*x;


	float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;
	
	float ray_x=(ray1[0])*width;
	float ray_y=(float)height-(ray1[1]*(float)height);
	ray_x=max(1.0f, min((float)width-2, ray_x ));
	ray_y=max(1.0f, min((float)height-2, ray_y ));

	float* before = tex2D(surfaceSrc, ray_x + .5f, ray_y + .5f, pitch);

	float* before1 = tex2D(surfaceDst1, ray_x + .5f, ray_y + .5f, pitch);
	float* before2 = tex2D(surfaceDst2, ray_x + .5f, ray_y + .5f, pitch);
	float* before3 = tex2D(surfaceDst3, ray_x + .5f, ray_y + .5f, pitch);

	float4 triangle[4];
	triangle[0].x = before1[0];
	triangle[0].y = before1[1];
	triangle[0].z = before1[2];
	triangle[0].w = before1[3];

	triangle[1].x = before2[0];
	triangle[1].y = before2[1];
	triangle[1].z = before2[2];
	triangle[1].w = before2[3];

	triangle[2].x = before3[0];
	triangle[2].y = before3[1];
	triangle[2].z = before3[2]+1.0f;
	triangle[2].w = before3[3];

	triangle[3].x = before3[0];
	triangle[3].y = before3[1];
	triangle[3].z = before3[2];
	triangle[3].w = before3[3];

	

	//map the shadow
	if(ray1[2] < before[3]){
		pixel[0]=before[0]*.5;
		pixel[1]=before[1]*.5;
		pixel[2]=before[2]*.5;
		pixel[3]=before[3]*.5;
	} else {
		pixel[0]=before[0];
		pixel[1]=before[1];
		pixel[2]=before[2];
		pixel[3]=before[3];
	}
}

__global__ void kernelPolygonMapBlend5(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* worldView;
	worldView = (float4*)(matrices+4*3);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
	__shared__ float4* worldViewLast;
	worldViewLast = (float4*)(matrices+4*15);
	__shared__ float4* invWorldViewLast;
	invWorldViewLast = (float4*)(matrices+4*21);
	__shared__ float4* invWorldViewProjLast;
	invWorldViewProjLast = (float4*)(matrices+4*23);

#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
			// get a pointer to the source ray and destination pixel at (x,y)
			float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
			float* pixel1 = (float*)(surfaceDst1 + y*pitch) + 4*x;
			float* pixel2 = (float*)(surfaceDst2 + y*pitch) + 4*x;
			float* pixel3 = (float*)(surfaceDst3 + y*pitch) + 4*x;

			float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
			float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;

			float ray_x=(ray1[0])*width;
			float ray_y=(float)height-(ray1[1]*(float)height);
			ray_x=max(1.0f, min((float)width-1, ray_x ));
			ray_y=max(1.0f, min((float)height-1, ray_y ));

			float* before = tex2D(surfaceSrc, ray_x, ray_y, pitch);

			float4 rayDir;
			float4 rayOrig;
			shadowRay(x, y, width, height, ray1, &rayDir, &rayOrig, projection, invWorldView, invWorldViewLast);

			float4* triangle00 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y, pitch);
			float dist00=intersectTriangle(triangle00, rayDir, rayOrig);
			float4* triangle01 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y+1.0f, pitch);
			float dist01=intersectTriangle(triangle01, rayDir, rayOrig);
			float4* triangle10 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y, pitch);
			float dist10=intersectTriangle(triangle10, rayDir, rayOrig);
			float4* triangle11 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y+1.0f, pitch);
			float dist11=intersectTriangle(triangle11, rayDir, rayOrig);

			//map the shadow
			//pixel[0]=before[0]+.8f;
			//pixel[1]=before[1]+.8f;
			//pixel[2]=before[2]+.8f;
			pixel[0]=1.0f;
			pixel[1]=1.0f;
			pixel[2]=1.0f;
			pixel[3]=.5f;
			
			//if( dist00 > 0.0 || dist01 > 0.0 || dist10 > 0.0 || dist11 > 0.0 ){
			//if( dist11 > Z_NEAR ){
			if( dist00 > Z_NEAR || dist01 > Z_NEAR || dist10 > Z_NEAR || dist11 > Z_NEAR ){
				pixel[0]*=.25f;
				pixel[1]*=.25f;
				pixel[2]*=.25f;
			}

			//if( dist00 > Z_NEAR ){
			//	pixel[2]=1.0f;
			//}
			//if( dist01 > Z_NEAR ){
			//	pixel[1]=1.0f;
			//}
			//if( dist11 > Z_NEAR ){
			//	pixel[0]=1.0f;
			//}
		}
	}

__global__ void kernelPolygonMapBlend6(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* worldView;
	worldView = (float4*)(matrices+4*3);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
	__shared__ float4* worldViewLast;
	worldViewLast = (float4*)(matrices+4*15);
	__shared__ float4* invWorldViewLast;
	invWorldViewLast = (float4*)(matrices+4*21);
	__shared__ float4* invWorldViewProjLast;
	invWorldViewProjLast = (float4*)(matrices+4*23);

#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
			// get a pointer to the source ray and destination pixel at (x,y)
			float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
			float* pixel1 = (float*)(surfaceDst1 + y*pitch) + 4*x;
			float* pixel2 = (float*)(surfaceDst2 + y*pitch) + 4*x;
			float* pixel3 = (float*)(surfaceDst3 + y*pitch) + 4*x;

			float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
			float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;

			float ray_x=(ray1[0])*width;
			float ray_y=(float)height-(ray1[1]*(float)height);
			ray_x=max(1.0f, min((float)width-1, ray_x ));
			ray_y=max(1.0f, min((float)height-1, ray_y ));

			float* before = tex2D(surfaceSrc, ray_x, ray_y, pitch);

			float4 rayDir;
			float4 rayOrig;
			shadowRay(x, y, width, height, ray1, &rayDir, &rayOrig, projection, invWorldView, invWorldViewLast);

			float4* triangle00 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y, pitch);
			float dist00=intersectTriangleZNear(triangle00, rayDir, rayOrig, 0.0005f );
			float4* triangle01 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y+1.0f, pitch);
			float dist01=intersectTriangleZNear(triangle01, rayDir, rayOrig, 0.0005f );
			float4* triangle10 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y, pitch);
			float dist10=intersectTriangleZNear(triangle10, rayDir, rayOrig, 0.0005f );
			float4* triangle11 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y+1.0f, pitch);
			float dist11=intersectTriangleZNear(triangle11, rayDir, rayOrig, 0.0005f );

			////map the shadow
			//pixel[0]=before[0]*2.8f;
			//pixel[1]=before[1]*2.8f;
			//pixel[2]=before[2]*2.8f;
			pixel[0]=1.0f;
			pixel[1]=1.0f;
			pixel[2]=1.0f;
			pixel[3]=.5f;
			
			if( dist00 > 0.0f || dist01 > 0.0f || dist10 > 0.0f || dist11 > 0.0f ){
			//if( dist00 > -Z_NEAR || dist01 > -Z_NEAR || dist10 > -Z_NEAR || dist11 > -Z_NEAR ){
				pixel[0]*=.2f;
				pixel[1]*=.2f;
				pixel[2]*=.2f;
			}
		}
	}

	__global__ void kernelPolygonMapBlend7(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
	float* ray = (float*)(surfaceSrc1 + y*pitch) + 4*x;
	
	float ray_x=(ray[0])*width;
	float ray_y=(float)height-(ray[1]*(float)height);
	ray_x=max(1.0f, min((float)width-2, ray_x + rand.x ));
	ray_y=max(1.0f, min((float)height-2, ray_y + rand.y ));

	pixel[0]=1.0f;
	pixel[1]=1.0f;
	pixel[2]=1.0f;
	pixel[3]=.5f;

	float* before = tex2D(surfaceSrc, ray_x, ray_y, pitch);
	float hit = before[3] - ray[2];
	before = tex2D(surfaceSrc, ray_x+1.0f, ray_y, pitch);
	hit = min(hit,  before[3] - ray[2]);
	before = tex2D(surfaceSrc, ray_x, ray_y+1.0f, pitch);
	hit = min(hit,  before[3] - ray[2]);
	before = tex2D(surfaceSrc, ray_x+1.0f, ray_y+1.0f, pitch);
	hit = min(hit,  before[3] - ray[2]);
	if( hit > 0.0002f ){
		pixel[0]*=.4f;
		pixel[1]*=.4f;
		pixel[2]*=.4f;
	}
}

__global__ void kernelPolygonMapBlend8(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* worldView;
	worldView = (float4*)(matrices+4*3);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
	__shared__ float4* worldViewLast;
	worldViewLast = (float4*)(matrices+4*15);
	__shared__ float4* invWorldViewLast;
	invWorldViewLast = (float4*)(matrices+4*21);
	__shared__ float4* invWorldViewProjLast;
	invWorldViewProjLast = (float4*)(matrices+4*23);

#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
			// get a pointer to the source ray and destination pixel at (x,y)
			float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
			float* pixel1 = (float*)(surfaceDst1 + y*pitch) + 4*x;
			float* pixel2 = (float*)(surfaceDst2 + y*pitch) + 4*x;
			float* pixel3 = (float*)(surfaceDst3 + y*pitch) + 4*x;

			float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
			float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;

			float ray_x=(ray1[0])*width;
			float ray_y=(float)height-(ray1[1]*(float)height);
			ray_x=max(1.0f, min((float)width-1, ray_x ));
			ray_y=max(1.0f, min((float)height-1, ray_y ));

			pixel[0]=1.0f;
			pixel[1]=1.0f;
			pixel[2]=1.0f;
			pixel[3]=1.0f;

			float* before = tex2D(surfaceSrc, ray_x, ray_y, pitch);
			float hit = before[3] - ray1[2];
			before = tex2D(surfaceSrc, ray_x+1.0f, ray_y, pitch);
			hit = min(hit,  before[3] - ray1[2]);
			before = tex2D(surfaceSrc, ray_x, ray_y+1.0f, pitch);
			hit = min(hit,  before[3] - ray1[2]);
			before = tex2D(surfaceSrc, ray_x+1.0f, ray_y+1.0f, pitch);
			hit = min(hit,  before[3] - ray1[2]);
			if(hit > 0.0002f ){
				pixel[0]*=.0f;
				pixel[1]*=.0f;
				pixel[2]*=.0f;
				return;
			}

			float4 rayDir;
			float4 rayOrig;
			shadowRay(x, y, width, height, ray1, &rayDir, &rayOrig, projection, invWorldView, invWorldViewLast);

			float4* triangle00 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y, pitch);
			float dist00=intersectTriangleZNear(triangle00, rayDir, rayOrig, 0.0005f );
			float4* triangle01 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y+1.0f, pitch);
			float dist01=intersectTriangleZNear(triangle01, rayDir, rayOrig, 0.0005f );
			float4* triangle10 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y, pitch);
			float dist10=intersectTriangleZNear(triangle10, rayDir, rayOrig, 0.0005f );
			float4* triangle11 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y+1.0f, pitch);
			float dist11=intersectTriangleZNear(triangle11, rayDir, rayOrig, 0.0005f );

			////map the shadow
			//pixel[0]=before[0]*2.8f;
			//pixel[1]=before[1]*2.8f;
			//pixel[2]=before[2]*2.8f;
			
			
			if( dist00 > 0.0f || dist01 > 0.0f || dist10 > 0.0f || dist11 > 0.0f ){
			//if( dist00 > -Z_NEAR || dist01 > -Z_NEAR || dist10 > -Z_NEAR || dist11 > -Z_NEAR ){
				pixel[0]*=.2f;
				pixel[1]*=.2f;
				pixel[2]*=.2f;
			}
		}
	}

__global__ void kernelPolygonMapBlend9(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* worldView;
	worldView = (float4*)(matrices+4*3);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
	__shared__ float4* worldViewLast;
	worldViewLast = (float4*)(matrices+4*15);
	__shared__ float4* invWorldViewLast;
	invWorldViewLast = (float4*)(matrices+4*21);
	__shared__ float4* invWorldViewProjLast;
	invWorldViewProjLast = (float4*)(matrices+4*23);

#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
			// get a pointer to the source ray and destination pixel at (x,y)
			float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
			float* pixel1 = (float*)(surfaceDst1 + y*pitch) + 4*x;
			float* pixel2 = (float*)(surfaceDst2 + y*pitch) + 4*x;
			float* pixel3 = (float*)(surfaceDst3 + y*pitch) + 4*x;

			float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
			float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;

			float ray_x=(ray1[0])*width;
			float ray_y=(float)height-(ray1[1]*(float)height);
			ray_x=max(1.0f, min((float)width-1, ray_x ));
			ray_y=max(1.0f, min((float)height-1, ray_y ));

			pixel[0]=1.0f;
			pixel[1]=1.0f;
			pixel[2]=1.0f;
			pixel[3]=1.0f;

			float* before = tex2D(surfaceSrc, ray_x, ray_y, pitch);
			float hit = before[3] - ray1[2];
			before = tex2D(surfaceSrc, ray_x+1.0f, ray_y, pitch);
			hit = min(hit,  before[3] - ray1[2]);
			before = tex2D(surfaceSrc, ray_x, ray_y+1.0f, pitch);
			hit = min(hit,  before[3] - ray1[2]);
			before = tex2D(surfaceSrc, ray_x+1.0f, ray_y+1.0f, pitch);
			hit = min(hit,  before[3] - ray1[2]);
			if(hit > 0.0f ){
				pixel[0]*=.2f;
				pixel[1]*=.2f;
				pixel[2]*=.2f;
				return;
			}

			float4 rayDir;
			float4 rayOrig;
			shadowRay(x, y, width, height, ray1, &rayDir, &rayOrig, projection, invWorldView, invWorldViewLast);

			float4* triangle00 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y, pitch);
			float dist00=intersectTriangleZNear(triangle00, rayDir, rayOrig, 0.0005f );
			float4* triangle01 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y+1.0f, pitch);
			float dist01=intersectTriangleZNear(triangle01, rayDir, rayOrig, 0.0005f );
			float4* triangle10 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y, pitch);
			float dist10=intersectTriangleZNear(triangle10, rayDir, rayOrig, 0.0005f );
			float4* triangle11 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y+1.0f, pitch);
			float dist11=intersectTriangleZNear(triangle11, rayDir, rayOrig, 0.0005f );

			////map the shadow
			//pixel[0]=before[0]*2.8f;
			//pixel[1]=before[1]*2.8f;
			//pixel[2]=before[2]*2.8f;
			
			
			if( dist00 > 0.0f || dist01 > 0.0f || dist10 > 0.0f || dist11 > 0.0f ){
			//if( dist00 > Z_NEAR || dist01 > Z_NEAR || dist10 > Z_NEAR || dist11 > Z_NEAR ){
				pixel[0]*=.2f;
				pixel[1]*=.2f;
				pixel[2]*=.2f;
			}
		}
	}

__global__ void kernelPolygonMapBlend10(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* worldView;
	worldView = (float4*)(matrices+4*3);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
	__shared__ float4* worldViewLast;
	worldViewLast = (float4*)(matrices+4*15);
	__shared__ float4* invWorldViewLast;
	invWorldViewLast = (float4*)(matrices+4*21);
	__shared__ float4* invWorldViewProjLast;
	invWorldViewProjLast = (float4*)(matrices+4*23);

#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
			// get a pointer to the source ray and destination pixel at (x,y)
			float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
			float* pixel1 = (float*)(surfaceDst1 + y*pitch) + 4*x;
			float* pixel2 = (float*)(surfaceDst2 + y*pitch) + 4*x;
			float* pixel3 = (float*)(surfaceDst3 + y*pitch) + 4*x;

			float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
			float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;

			float ray_x=(ray1[0])*width;
			float ray_y=(float)height-(ray1[1]*(float)height);
			ray_x=max(1.0f, min((float)width-1, ray_x ));
			ray_y=max(1.0f, min((float)height-1, ray_y ));		

			float4 rayDir;
			float4 rayOrig;
			shadowRay(x, y, width, height, ray1, &rayDir, &rayOrig, projection, invWorldView, invWorldViewLast);

			float4* triangle00 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y, pitch);
			float dist00=intersectTriangleZNear(triangle00, rayDir, rayOrig, 0.0005f );
			float4* triangle01 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y+1.0f, pitch);
			float dist01=intersectTriangleZNear(triangle01, rayDir, rayOrig, 0.0005f );
			float4* triangle10 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y, pitch);
			float dist10=intersectTriangleZNear(triangle10, rayDir, rayOrig, 0.0005f );
			float4* triangle11 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y+1.0f, pitch);
			float dist11=intersectTriangleZNear(triangle11, rayDir, rayOrig, 0.0005f );
			
			////map the previous triangle color
			float* before = tex2D(surfaceSrc, ray_x, ray_y, pitch);
			if(dist00 < dist01) {
				dist00=dist01;
				before = tex2D(surfaceSrc, ray_x, ray_y + 1.0, pitch);
			}
			if(dist00 < dist10) {
				dist00=dist10;
				before = tex2D(surfaceSrc, ray_x + 1.0, ray_y, pitch);
			}
			if(dist00 < dist11) {
				before = tex2D(surfaceSrc, ray_x + 1.0, ray_y + 1.0, pitch);
			}
			
			pixel[0]=before[0];
			pixel[1]=before[1];
			pixel[2]=before[2];
			
			float px = floorf(ray_x);   // integer position
			float py = floorf(ray_y);
			float fx = ray_x - px;      // fractional position
			float fy = ray_y - py;    			
			
			if( fx < 0.05 || fy < 0.05 ){
				pixel[0]=0.5f;
				pixel[1]=0.5f;
				pixel[2]=0.5f;
			}
		}
	}
	
__global__ void kernelPolygonMapBlend11(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* worldView;
	worldView = (float4*)(matrices+4*3);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
	__shared__ float4* worldViewLast;
	worldViewLast = (float4*)(matrices+4*15);
	__shared__ float4* invWorldViewLast;
	invWorldViewLast = (float4*)(matrices+4*21);
	__shared__ float4* invWorldViewProjLast;
	invWorldViewProjLast = (float4*)(matrices+4*23);

#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
			// get a pointer to the source ray and destination pixel at (x,y)
			float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
			float* pixel1 = (float*)(surfaceDst1 + y*pitch) + 4*x;
			float* pixel2 = (float*)(surfaceDst2 + y*pitch) + 4*x;
			float* pixel3 = (float*)(surfaceDst3 + y*pitch) + 4*x;

			float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
			float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;

			float ray_x=(ray1[0])*width;
			float ray_y=(float)height-(ray1[1]*(float)height);
			ray_x=max(1.0f, min((float)width-1, ray_x ));
			ray_y=max(1.0f, min((float)height-1, ray_y ));

			float* before = tex2D(surfaceSrc, ray_x, ray_y, pitch);

			float4 rayDir;
			float4 rayOrig;
			shadowRay(x, y, width, height, ray1, &rayDir, &rayOrig, projection, invWorldView, invWorldViewLast);

			float4* triangle00 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y, pitch);
			float dist00=intersectTriangleZNear(triangle00, rayDir, rayOrig, 0.0005f );
			float4* triangle01 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y+1.0f, pitch);
			float dist01=intersectTriangleZNear(triangle01, rayDir, rayOrig, 0.0005f );
			float4* triangle10 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y, pitch);
			float dist10=intersectTriangleZNear(triangle10, rayDir, rayOrig, 0.0005f );
			float4* triangle11 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y+1.0f, pitch);
			float dist11=intersectTriangleZNear(triangle11, rayDir, rayOrig, 0.0005f );

			////map the shadow
			//pixel[0]=before[0]*2.8f;
			//pixel[1]=before[1]*2.8f;
			//pixel[2]=before[2]*2.8f;
			pixel[0]=1.0f;
			pixel[1]=1.0f;
			pixel[2]=1.0f;
			pixel[3]=.5f;
			
			if( dist00 > 0.001f || dist01 > 0.001f || dist10 > 0.001f || dist11 > 0.001f ){
			//if( dist00 > -Z_NEAR || dist01 > -Z_NEAR || dist10 > -Z_NEAR || dist11 > -Z_NEAR ){
				pixel[0]*=.2f;
				pixel[1]*=.2f;
				pixel[2]*=.2f;
			}
			
			float px = floorf(ray_x);   // integer position
			float py = floorf(ray_y);
			float fx = ray_x - px;      // fractional position
			float fy = ray_y - py;    			
			
			shadowRay(px, py, width, height, ray1, &rayDir, &rayOrig, projection, invWorldView, invWorldViewLast);
			triangle00 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, px, py, pitch);
			dist00=intersectTriangleZNear(triangle00, rayDir, rayOrig, 0.0005f );
			triangle01 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, px, py+1.0f, pitch);
			dist01=intersectTriangleZNear(triangle01, rayDir, rayOrig, 0.0005f );
			triangle10 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, px+1.0f, py, pitch);
			dist10=intersectTriangleZNear(triangle10, rayDir, rayOrig, 0.0005f );
			triangle11 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, px+1.0f, py+1.0f, pitch);
			dist11=intersectTriangleZNear(triangle11, rayDir, rayOrig, 0.0005f );
			
			//if( dist00 > 0.0f ) {
			if( dist00 > 0.0f || dist01 > 0.0f || dist10 > 0.0f || dist11 > 0.0f ){
			//if( dist00 > -Z_NEAR || dist01 > -Z_NEAR || dist10 > -Z_NEAR || dist11 > -Z_NEAR ){
				pixel[2]=0.5f;
			}
			
			if(fx < 0.05 || fy < 0.05){
				pixel[0]=0.5f;
			}
		}
	}

__global__ void kernelPolygonMapBlend12(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* worldView;
	worldView = (float4*)(matrices+4*3);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
	__shared__ float4* worldViewLast;
	worldViewLast = (float4*)(matrices+4*15);
	__shared__ float4* invWorldViewLast;
	invWorldViewLast = (float4*)(matrices+4*21);
	__shared__ float4* invWorldViewProjLast;
	invWorldViewProjLast = (float4*)(matrices+4*23);

#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
			// get a pointer to the source ray and destination pixel at (x,y)
			float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
			float* pixel1 = (float*)(surfaceDst1 + y*pitch) + 4*x;
			float* pixel2 = (float*)(surfaceDst2 + y*pitch) + 4*x;
			float* pixel3 = (float*)(surfaceDst3 + y*pitch) + 4*x;

			float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
			float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;

			float ray_x=(ray1[0])*width;
			float ray_y=(float)height-(ray1[1]*(float)height);
			ray_x=max(1.0f, min((float)width, ray_x ));
			ray_y=max(1.0f, min((float)height, ray_y ));		

			float4 rayDir;
			float4 rayOrig;

			float px = floorf(ray_x);   // integer position
			float py = floorf(ray_y);
			float fx = ray_x - px;      // fractional position
			float fy = ray_y - py;    	
						
			shadowRay(x, y, width, height, ray1, &rayDir, &rayOrig, projection, invWorldView, invWorldViewLast);

			float4* triangle00 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y, pitch);
			float t00;
			float dist00=intersectTriangleZNear(triangle00, rayDir, rayOrig, &t00, 0.0005f );
			float4* triangle01 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y+1.0f, pitch);
			float t01;
			float dist01=intersectTriangleZNear(triangle01, rayDir, rayOrig, &t01, 0.0005f );
			float4* triangle10 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y, pitch);
			float t10;
			float dist10=intersectTriangleZNear(triangle10, rayDir, rayOrig, &t10, 0.0005f );
			float4* triangle11 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y+1.0f, pitch);
			float t11;
			float dist11=intersectTriangleZNear(triangle11, rayDir, rayOrig, &t11, 0.0005f );			

			float* before = tex2D(surfaceSrc, ray_x+.5f, ray_y+.5f, pitch);
			pixel[0]=before[0]*2.0f;
			pixel[1]=before[1]*2.0f;
			pixel[2]=before[2]*2.0f;	
				
			
			if( t00 > Z_NEAR && t01 > Z_NEAR && t10 > Z_NEAR && t11 > Z_NEAR ){
				pixel[0]=min(1.0f, max(0.0f, 1.0f*max(t00, max(t01, max(t10, t11)))));
				pixel[1]=min(1.0f, max(0.0f, 1.0f*max(t00, max(t01, max(t10, t11)))));
				pixel[2]=min(1.0f, max(0.0f, 1.0f*max(t00, max(t01, max(t10, t11)))));
			}
			else if( dist00 > 0.0f || dist01 > 0.0f || dist10 > 0.0f || dist11 > 0.0f ){	
				pixel[0]=min(1.0f, max(0.0f, 1.0f*min(t00, min(t01, min(t10, t11)))));
				pixel[1]=min(1.0f, max(0.0f, 1.0f*min(t00, min(t01, min(t10, t11)))));
				pixel[2]=min(1.0f, max(0.0f, 1.0f*min(t00, min(t01, min(t10, t11)))));
			}
			
			if( fx < 0.05 || fy < 0.05 ){
				pixel[0]*=0.8f;
				pixel[1]*=0.8f;
				pixel[2]*=0.8f;
				pixel[0]+=0.1f;
				pixel[1]+=0.1f;
				pixel[2]+=0.1f;
			}
		}
	}

	__global__ void kernelPolygonMapBlend13(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* worldView;
	worldView = (float4*)(matrices+4*3);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
	__shared__ float4* worldViewLast;
	worldViewLast = (float4*)(matrices+4*15);
	__shared__ float4* invWorldViewLast;
	invWorldViewLast = (float4*)(matrices+4*21);
	__shared__ float4* invWorldViewProjLast;
	invWorldViewProjLast = (float4*)(matrices+4*23);

#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
			// get a pointer to the source ray and destination pixel at (x,y)
			float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
			float* pixel1 = (float*)(surfaceDst1 + y*pitch) + 4*x;
			float* pixel2 = (float*)(surfaceDst2 + y*pitch) + 4*x;
			float* pixel3 = (float*)(surfaceDst3 + y*pitch) + 4*x;

			float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
			float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;

			float ray_x=(ray1[0])*width;
			float ray_y=(float)height-(ray1[1]*(float)height);
			ray_x=max(1.0f, min((float)width, ray_x ));
			ray_y=max(1.0f, min((float)height, ray_y ));		

			float4 rayDir;
			float4 rayOrig;

			float px = floorf(ray_x);   // integer position
			float py = floorf(ray_y);
			float fx = ray_x - px;      // fractional position
			float fy = ray_y - py;    	
						
			shadowRay(x, y, width, height, ray1, &rayDir, &rayOrig, projection, invWorldView, invWorldViewLast);

			float4* triangle00 = tex2DTriangleMatrixZOnly(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y, pitch);
			float t00;
			intersectTriangleZOnly(triangle00, rayDir, rayOrig, &t00, 0.000f );
			float4* triangle01 = tex2DTriangleMatrixZOnly(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y+1.0f, pitch);
			float t01;
			intersectTriangleZOnly(triangle01, rayDir, rayOrig, &t01, 0.000f );
			float4* triangle10 = tex2DTriangleMatrixZOnly(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y, pitch);
			float t10;
			intersectTriangleZOnly(triangle10, rayDir, rayOrig, &t10, 0.000f );
			float4* triangle11 = tex2DTriangleMatrixZOnly(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y+1.0f, pitch);
			float t11;
			intersectTriangleZOnly(triangle11, rayDir, rayOrig, &t11, 0.000f );			

			float* before = tex2D(surfaceSrc, ray_x+.5f, ray_y+.5f, pitch);
			pixel[0]=before[0]*2.0f;
			pixel[1]=before[1]*2.0f;
			pixel[2]=before[2]*2.0f;	
				
			
			if( t00 > Z_NEAR && t01 > Z_NEAR && t10 > Z_NEAR && t11 > Z_NEAR ){
				pixel[0]*=0.1f;
				pixel[1]*=0.1f;
				pixel[2]*=0.1f;
			}
			
			if( fx < 0.05 || fy < 0.05 ){
				pixel[0]*=0.8f;
				pixel[1]*=0.8f;
				pixel[2]*=0.8f;
				pixel[0]+=0.1f;
				pixel[1]+=0.1f;
				pixel[2]+=0.1f;
			}
		}
	}

__global__ void kernelPolygonMapBlend14(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* worldView;
	worldView = (float4*)(matrices+4*3);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
	__shared__ float4* worldViewLast;
	worldViewLast = (float4*)(matrices+4*15);
	__shared__ float4* invWorldViewLast;
	invWorldViewLast = (float4*)(matrices+4*21);
	__shared__ float4* invWorldViewProjLast;
	invWorldViewProjLast = (float4*)(matrices+4*23);

#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
			// get a pointer to the source ray and destination pixel at (x,y)
			float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
			float* pixel1 = (float*)(surfaceDst1 + y*pitch) + 4*x;
			float* pixel2 = (float*)(surfaceDst2 + y*pitch) + 4*x;
			float* pixel3 = (float*)(surfaceDst3 + y*pitch) + 4*x;

			float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
			float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;

			float ray_x=(ray1[0])*width;
			float ray_y=(float)height-(ray1[1]*(float)height);
			ray_x=max(1.0f, min((float)width, ray_x ));
			ray_y=max(1.0f, min((float)height, ray_y ));		

			float4 rayDir;
			float4 rayOrig;

			float px = floorf(ray_x);   // integer position
			float py = floorf(ray_y);
			float fx = ray_x - px;      // fractional position
			float fy = ray_y - py;    	
						
			shadowRay(x, y, width, height, ray1, &rayDir, &rayOrig, projection, invWorldView, invWorldViewLast);

			float4* triangle00 = tex2DTriangleMatrixZOnly(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+.5f, ray_y+.5f, pitch);
			float t00;
			intersectTriangleZOnly(triangle00, rayDir, rayOrig, &t00, 0.0f );

			float* before = tex2D(surfaceSrc, ray_x+.5f, ray_y+.5f, pitch);
			pixel[0]=before[0]*2.0f;
			pixel[1]=before[1]*2.0f;
			pixel[2]=before[2]*2.0f;		
			
			if( t00 > Z_NEAR ){
				pixel[0]*=0.1f;
				pixel[1]*=0.1f;
				pixel[2]*=0.1f;
			}
			if( fx < 0.05 || fy < 0.05 ){
				pixel[0]*=0.8f;
				pixel[1]*=0.8f;
				pixel[2]*=0.8f;
				pixel[0]+=0.1f;
				pixel[1]+=0.1f;
				pixel[2]+=0.1f;
			}
		}
	}

__global__ void kernelPolygonMapBlend15(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* worldView;
	worldView = (float4*)(matrices+4*3);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
	__shared__ float4* worldViewLast;
	worldViewLast = (float4*)(matrices+4*15);
	__shared__ float4* invWorldViewLast;
	invWorldViewLast = (float4*)(matrices+4*21);
	__shared__ float4* invWorldViewProjLast;
	invWorldViewProjLast = (float4*)(matrices+4*23);

#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
			// get a pointer to the source ray and destination pixel at (x,y)
			float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
			float* pixel1 = (float*)(surfaceDst1 + y*pitch) + 4*x;
			float* pixel2 = (float*)(surfaceDst2 + y*pitch) + 4*x;
			float* pixel3 = (float*)(surfaceDst3 + y*pitch) + 4*x;

			float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
			float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;

			float ray_x=(ray1[0])*width;
			float ray_y=(float)height-(ray1[1]*(float)height);
			ray_x=max(1.0f, min((float)width, ray_x ));
			ray_y=max(1.0f, min((float)height, ray_y ));		

			float4 rayDir;
			float4 rayOrig;

			float px = floorf(ray_x);   // integer position
			float py = floorf(ray_y);
			float fx = ray_x - px;      // fractional position
			float fy = ray_y - py;    	
						
			shadowRay(x, y, width, height, ray1, &rayDir, &rayOrig, projection, invWorldView, invWorldViewLast);

						float4* triangle00 = tex2DTriangleMatrixZOnly(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y, pitch);
			float t00;
			intersectTriangleZOnly(triangle00, rayDir, rayOrig, &t00, 0.000f );
			float4* triangle01 = tex2DTriangleMatrixZOnly(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y+1.0f, pitch);
			float t01;
			intersectTriangleZOnly(triangle01, rayDir, rayOrig, &t01, 0.000f );
			float4* triangle10 = tex2DTriangleMatrixZOnly(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y, pitch);
			float t10;
			intersectTriangleZOnly(triangle10, rayDir, rayOrig, &t10, 0.000f );
			float4* triangle11 = tex2DTriangleMatrixZOnly(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y+1.0f, pitch);
			float t11;
			intersectTriangleZOnly(triangle11, rayDir, rayOrig, &t11, 0.000f );			

			float* before = tex2D(surfaceSrc, ray_x+.5f, ray_y+.5f, pitch);
			pixel[0]=1.0f;
			pixel[1]=1.0f;
			pixel[2]=1.0f;	
			
			//if( t00 > 0.0001 && t01 > 0.0001 && t10 > 0.0001 && t11 > 0.0001 ){
				pixel[0]=min(1.0f, max(0.0f, 1.0f-1.0f*min(t00, min(t01, min(t10, t11)))));
				pixel[1]=min(1.0f, max(0.0f, 1.0f-1.0f*min(t00, min(t01, min(t10, t11)))));
				pixel[2]=min(1.0f, max(0.0f, 1.0f-1.0f*min(t00, min(t01, min(t10, t11)))));
			//}
			if( fx < 0.05 || fy < 0.05 ){
				pixel[0]*=0.8f;
				pixel[1]*=0.8f;
				pixel[2]*=0.8f;
				pixel[0]+=0.1f;
				pixel[1]+=0.1f;
				pixel[2]+=0.1f;
			}
		}
	}

__global__ void kernelPolygonMapBlend16(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* worldView;
	worldView = (float4*)(matrices+4*3);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
	__shared__ float4* worldViewLast;
	worldViewLast = (float4*)(matrices+4*15);
	__shared__ float4* invWorldViewLast;
	invWorldViewLast = (float4*)(matrices+4*21);
	__shared__ float4* invWorldViewProjLast;
	invWorldViewProjLast = (float4*)(matrices+4*23);

#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
			// get a pointer to the source ray and destination pixel at (x,y)
			float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
			float* pixel1 = (float*)(surfaceDst1 + y*pitch) + 4*x;
			float* pixel2 = (float*)(surfaceDst2 + y*pitch) + 4*x;
			float* pixel3 = (float*)(surfaceDst3 + y*pitch) + 4*x;

			float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
			float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;

			float ray_x=(ray1[0])*width;
			float ray_y=(float)height-(ray1[1]*(float)height);
			ray_x=max(1.0f, min((float)width, ray_x ));
			ray_y=max(1.0f, min((float)height, ray_y ));		

			float4 rayDir;
			float4 rayOrig;

			float px = floorf(ray_x);   // integer position
			float py = floorf(ray_y);
			float fx = ray_x - px;      // fractional position
			float fy = ray_y - py;    	
						
			shadowRay(x, y, width, height, ray1, &rayDir, &rayOrig, projection, invWorldView, invWorldViewLast);

						float4* triangle00 = tex2DTriangleMatrixZOnly(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y, pitch);
			float t00;
			intersectTriangleZOnly(triangle00, rayDir, rayOrig, &t00, 0.000f );
			float4* triangle01 = tex2DTriangleMatrixZOnly(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y+1.0f, pitch);
			float t01;
			intersectTriangleZOnly(triangle01, rayDir, rayOrig, &t01, 0.000f );
			float4* triangle10 = tex2DTriangleMatrixZOnly(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y, pitch);
			float t10;
			intersectTriangleZOnly(triangle10, rayDir, rayOrig, &t10, 0.000f );
			float4* triangle11 = tex2DTriangleMatrixZOnly(surfaceDst1, surfaceDst2, surfaceDst3, ray_x+1.0f, ray_y+1.0f, pitch);
			float t11;
			intersectTriangleZOnly(triangle11, rayDir, rayOrig, &t11, 0.000f );			

			float* before = tex2D(surfaceSrc, ray_x+.5f, ray_y+.5f, pitch);
			pixel[0]=1.0f;
			pixel[1]=1.0f;
			pixel[2]=1.0f;	

			if( t00 > 0.0001 && t01 > 0.0001 && t10 > 0.0001 && t11 > 0.0001 && fx < 0.25 && fy < 0.25){
				pixel[0]=0;
				pixel[1]=0;
				pixel[2]=0;
			}

			
			////if( t00 > 0.0001 && t01 > 0.0001 && t10 > 0.0001 && t11 > 0.0001 ){
			//	pixel[0]=min(1.0f, max(0.0f, 1.0f-1.0f*min(t00, min(t01, min(t10, t11)))));
			//	pixel[1]=min(1.0f, max(0.0f, 1.0f-1.0f*min(t00, min(t01, min(t10, t11)))));
			//	pixel[2]=min(1.0f, max(0.0f, 1.0f-1.0f*min(t00, min(t01, min(t10, t11)))));
			////}
			//if( fx < 0.05 || fy < 0.05 ){
			//	pixel[0]*=0.8f;
			//	pixel[1]*=0.8f;
			//	pixel[2]*=0.8f;
			//	pixel[0]+=0.1f;
			//	pixel[1]+=0.1f;
			//	pixel[2]+=0.1f;
			//}
		}
	}

	__global__ void kernelPolygonMapBlend17(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* worldView;
	worldView = (float4*)(matrices+4*3);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
	__shared__ float4* worldViewLast;
	worldViewLast = (float4*)(matrices+4*15);
	__shared__ float4* invWorldViewLast;
	invWorldViewLast = (float4*)(matrices+4*21);
	__shared__ float4* invWorldViewProjLast;
	invWorldViewProjLast = (float4*)(matrices+4*23);

#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
			// get a pointer to the source ray and destination pixel at (x,y)
			float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
			float* pixel1 = (float*)(surfaceDst1 + y*pitch) + 4*x;
			float* pixel2 = (float*)(surfaceDst2 + y*pitch) + 4*x;
			float* pixel3 = (float*)(surfaceDst3 + y*pitch) + 4*x;

			float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
			float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;

			float ray_x=(ray1[0])*width;
			float ray_y=(float)height-(ray1[1]*(float)height);
			ray_x=max(1.0f, min((float)width, ray_x));
			ray_y=max(1.0f, min((float)height, ray_y));		

			float4 rayDir;
			float4 rayOrig;

			float px = floorf(ray_x);   // integer position
			float py = floorf(ray_y);
			float fx = ray_x - px;      // fractional position
			float fy = ray_y - py;    	
						
			shadowRay(x, y, width, height, ray1, &rayDir, &rayOrig, projection, invWorldView, invWorldViewLast);

			float4* triangle00 = tex2DTriangleMatrix(surfaceDst1, surfaceDst2, surfaceDst3, ray_x, ray_y, pitch);
			float t00;
			float dist00=intersectTriangleZNear(triangle00, rayDir, rayOrig, &t00, 0.000f );

			float* before = tex2D(surfaceSrc, ray_x, ray_y, pitch);
			float* beforePixel = tex2D(surfaceDst, ray_x, ray_y, pitch);
			//if(dist00>0.0005f){
				beforePixel[0]=1.0f;
				beforePixel[1]=1.0f;
				beforePixel[2]=0.5f;
			//}
			//if(beforePixel[3] < triangle00[3].z) {
			//	beforePixel[0]=ray[0];
			//	beforePixel[1]=ray[1];
			//	beforePixel[2]=ray[2];
			//	beforePixel[3]=ray[3];
			//}
			

			//if( t00 > 0.0001 && t01 > 0.0001 && t10 > 0.0001 && t11 > 0.0001 && fx < 0.25 && fy < 0.25){
			//	beforePixel[0]=0;
			//	beforePixel[1]=0;
			//	beforePixel[2]=0;
			//}
		}
	}

__global__ void kernelBacktrackFixBlend(unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
	float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
	float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;
	
	float ray_x=(ray[0])*width;
	float ray_y=(float)height-(ray[1]*(float)height);
	ray_x=max(1.0f, min((float)width-2, ray_x));
	ray_y=max(1.0f, min((float)height-2, ray_y));

	float4 before = tex2DBilinear(surfaceSrc1, ray_x, ray_y, pitch);
	//float4 color = tex2DBilinear(surfaceSrc, ray_x, ray_y, pitch);
	//float* before = tex2D(surfaceSrc1, ray_x, ray_y, pitch);

	//pixel[0]=0.5f;
	//pixel[1]=0.5f;
	//pixel[2]=0.5f;

	//map the shadow
	if(ray[2] < before.w){
		pixel[0]=ray1[0];
		pixel[1]=ray1[1];
		pixel[2]=ray1[2];
	} else {
		//pixel[0]=1.0f;
		//pixel[1]=1.0f;
		//pixel[2]=1.0f;
	}

	//float dim=(1.0f/before.w-1.0f/ray[2]);
	//pixel[0]=dim;
	//pixel[1]=dim;
	//pixel[2]=dim;
}



__global__ void kernelStencilMapBlend(unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
	float* ray1 = (float*)(surfaceSrc + y*pitch) + 4*x;
	float* ray = (float*)(surfaceSrc1 + y*pitch) + 4*x;
	
	
	float ray_x=(ray[0])*width;
	float ray_y=(float)height-(ray[1]*(float)height);
	ray_x=max(1.0f, min((float)width-2, ray_x));
	ray_y=max(1.0f, min((float)height-2, ray_y));

	float4 before = tex2DBilinear(surfaceSrc, ray_x, ray_y, pitch);
	//float* before = tex2D(surfaceSrc, ray_x, ray_y, pitch);

	float4 rayOrig;
	rayOrig.x=0.0f;
	rayOrig.y=0.0f;
	rayOrig.z=2.0f;
	rayOrig.w=1.0f;

	float4 rayDir;
	rayDir.x=0.0f;
	rayDir.y=0.0f;
	rayDir.z=1.0f;
	rayDir.w=0.0f;

	float4 intersectionFront = tex2DPlanesIntersections(surfaceSrc2, before.x, before.y, pitch, rayOrig, rayDir);
	float4 intersectionBack = tex2DPlanesIntersections(surfaceSrc3, before.x, before.y, pitch, rayOrig, rayDir);		

	float4 intersection;
	intersection.x=intersectionBack.x-intersectionFront.x;
	intersection.y=intersectionBack.y-intersectionFront.y;
	intersection.z=intersectionBack.z-intersectionFront.z;
	intersection.w=intersectionBack.w-intersectionFront.w;

	float hasShadow = 
		max( intersection.x, 
		max( intersection.y, 
		max( intersection.z, intersection.w)));
	
	if(hasShadow>0.0f)
		hasShadow=1.0f;
	else
		hasShadow=0.0f;

	pixel[0]=intersectionBack.x;
	pixel[1]=hasShadow;
	pixel[2]=hasShadow;

	//pixel[0]=-intersection.x;
	//pixel[1]=-intersection.y;
	//pixel[2]=-intersection.z;

	//float hasShadow=  (ray[2] - before.w);
	//pixel[0]=hasShadow*1000.0f+1.0f;
	//pixel[1]=hasShadow*1000.0f+1.0f;
	//pixel[2]=hasShadow*1000.0f+1.0f;

	////map the shadow
	//if( ray[2] < before[3] ){
	//	pixel[0]=0.0f;
	//	pixel[1]=0.0f;
	//	pixel[2]=0.0f;
	//} else {
	//	pixel[0]=1.0f;
	//	pixel[1]=1.0f;
	//	pixel[2]=1.0f;
	//}

	//float dim=(1.0f/before.w-1.0f/ray[2]);
	//pixel[0]=dim;
	//pixel[1]=dim;
	//pixel[2]=dim;
}


//__global__ void kernelStencilMapBlend(unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
//{
//	//read the needed matrices for projection
//	__shared__ float4* projection;
//	projection = (float4*)(matrices+4*2);
//	__shared__ float4* invWorldView;
//	invWorldView = (float4*)(matrices+4*9);
//
//    int x = blockIdx.x*blockDim.x + threadIdx.x;
//    int y = blockIdx.y*blockDim.y + threadIdx.y;
//       
//    // in the case where, due to quantization into grids, we have
//    // more threads than pixels, skip the threads which don't 
//    // correspond to valid pixels
//	if (x >= width || y >= height) return;
//
//	// get a pointer to the source ray and destination pixel at (x,y)
//	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
//	float* ray = (float*)(surfaceSrc1 + y*pitch) + 4*x;
//	//float* ray1 = (float*)(surfaceSrc2 + y*pitch) + 4*x;
//	
//	float ray_x=(ray[0])*width;
//	float ray_y=(float)height-(ray[1]*(float)height);
//	ray_x=max(1.0f, min((float)width-2, ray_x));
//	ray_y=max(1.0f, min((float)height-2, ray_y));
//
//	float4 rayOrig;
//	rayOrig.x=0.0f;
//	rayOrig.y=0.0f;
//	rayOrig.z=2.0f;
//	rayOrig.w=1.0f;
//
//	float4 rayDir;
//	rayDir.x=0.0f;
//	rayDir.y=0.0f;
//	rayDir.z=1.0f;
//	rayDir.w=0.0f;
//
//	//float4 rayDir;
//	//float4 rayOrig;
//	//castRay(x, y, width, height, &rayDir, &rayOrig, projection, invWorldView);
//
//	float4 before = tex2DBilinear(surfaceSrc1, ray_x, ray_y, pitch);
//
//	//float4 rayDir;
//	//float4 rayOrig;
//	//reflectRay(x, y, width, height, pixel, &rayDir, &rayOrig, &before, projection, invWorldView);	
//	
//	float4 intersectionFront = tex2DPlanesIntersections(surfaceSrc2, before.x, before.y, pitch, rayOrig, rayDir);
//	float4 intersectionBack = tex2DPlanesIntersections(surfaceSrc3, before.x, before.y, pitch, rayOrig, rayDir);		
//
//	float4 intersection;
//	intersection.x=intersectionFront.x-intersectionBack.x;
//	intersection.y=intersectionFront.y-intersectionBack.y;
//	intersection.z=intersectionFront.z-intersectionBack.z;
//	intersection.w=intersectionFront.w-intersectionBack.w;
//
//	//float hasShadow = 
//	//	max( intersection.x, 
//	//	max( intersection.y, 
//	//	max( intersection.z, intersection.w)));
//
//	//pixel[0]=hasShadow;
//	//pixel[1]=hasShadow;
//	//pixel[2]=hasShadow;
//
//	pixel[0]=intersection.x;
//	pixel[1]=intersection.y;
//	pixel[2]=intersection.z;
//
//
//	//float hasShadow = intersectionBack.x - before.w;
//	//float hasShadow = intersectionFront.x - max(before.w, intersectionBack.x);
//	//float hasShadow = intersectionFront.x - intersectionBack.x;
//
//	//if(hasShadow>0.0) {
//	//	pixel[0]=0.0f;
//	//	pixel[1]=0.0f;
//	//	pixel[2]=0.0f;
//	//} else {
//	//	pixel[0]=1.0f;
//	//	pixel[1]=1.0f;
//	//	pixel[2]=1.0f;
//	//}
//}


__global__ void kernelTrackTextureBlend(unsigned char* surfaceSrc, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
       
    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't 
    // correspond to valid pixels
	if (x >= width || y >= height) return;

	// get a pointer to the source ray and destination pixel at (x,y)
	float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
	float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
	
	float ray_x=(ray[0])*width;
	float ray_y=(1.0f-ray[1])*height;
	ray_x=max(1.0f, min((float)width-2, ray_x));
	ray_y=max(1.0f, min((float)height-2, ray_y));

	float4 before = tex2DBilinear(surfaceDst, ray_x, ray_y, pitch);

	//if(ray[2] >= 1.0f  || ray[3]>=1.0){
	if(ray[3]==0.0){
		pixel[0]=pixel[0]*(1.0f-rand.w)+(.5*rand.w);
		pixel[1]=pixel[1]*(1.0f-rand.w)+(.5*rand.w);
		pixel[2]=pixel[2]*(1.0f-rand.w)+(.5*rand.w);
		return;
	}

	pixel[0]=before.x;
	pixel[1]=before.y;
	pixel[2]=before.z;
}


__global__ void kernelReflectBlend(unsigned char* surfaceSrc, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
		// get a pointer to the source ray and destination pixel at (x,y)
		float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
		float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
		float4 rayDir;
		float4 rayOrig;
		//make uninvolved pixels pint in lightdir
#ifdef SCANLINE_MASK						
		if(ray[3]==0.0f) {
			castRay(x, y, width, height, &rayDir, &rayOrig, projection, invWorldView);
			pixel[0] = rayDir.x*.25+.5;
			pixel[1] = rayDir.y*.25+.5;
			pixel[2] = rayDir.z*.25+.5;
			pixel[3] = rayDir.w;
#ifdef CUDA_RAYPOOL
			continue;
#endif
#ifndef CUDA_RAYPOOL
			return;
#endif
		}
#endif				
		float4 raySrf;
		reflectRayBias(x, y, width, height, ray, &rayDir, &rayOrig, &raySrf, projection, invWorldView);
				
		//raySrf = mulMatVec(invWorldView, raySrf);		

		float4* matrix=traceReflection(rayDir, rayOrig, tris, splitPlanes, splitIndices);
		if(matrix) {
			pixel[0] = (pixel[0]*(1.0f-rand.w))+((matrix[0].w+1.0f)*.5f*rand.w); // red
			pixel[1] = (pixel[1]*(1.0f-rand.w))+((matrix[1].w+1.0f)*.5f*rand.w); // green
			pixel[2] = (pixel[2]*(1.0f-rand.w))+((matrix[2].w+1.0f)*.5f*rand.w); // blue
			pixel[3] = (pixel[3]*(1.0f-rand.w))+((matrix[3].w+1.0f)*.5f*rand.w); // blue	
		} else {
			pixel[0] = raySrf.x*.5f+.5f; // red
			pixel[1] = raySrf.y*.5f+.5f; // green
			pixel[2] = raySrf.z*.5f+.5f; // blue
			pixel[3] = raySrf.w;
		}
	}
};
		

// global variables
__global__ void kernelRefractBlend(unsigned char* surfaceSrc, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif


		// get a pointer to the source ray and destination pixel at (x,y)
		float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
		float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
		float4 rayDir;
		float4 rayOrig;
		//make uninvolved pixels pint in lightdir
#ifdef SCANLINE_MASK				
		if(ray[3]==0.0f) {
			castRay(x, y, width, height, &rayDir, &rayOrig, projection, invWorldView);
			pixel[0] = rayDir.x*.25+.5;
			pixel[1] = rayDir.y*.25+.5;
			pixel[2] = rayDir.z*.25+.5;
			pixel[3] = rayDir.w;
#ifdef CUDA_RAYPOOL
			continue;
#endif
#ifndef CUDA_RAYPOOL
			return;
#endif
		}
#endif

		__shared__ float4 light;
		light.x=0.0f;
		light.y=0.0f;
		light.z=1.0f;
		light.w=0.0f;

		float4 raySrf;
		refractRayBias(x, y, width, height, ray, &rayDir, &rayOrig, &raySrf, projection, invWorldView, 0.5f, 1.0f);
		
		float l=dot(light,raySrf)+GOOCHY;
		//pixel[0] = l*LIGHT; // red
		//pixel[1] = l*LIGHT; // green
		//pixel[2] = l*LIGHT; // blue
		//pixel[3] = 0.5; // alpha, set to Zero for initial AO
		
		raySrf = mulMatVec(invWorldView, raySrf);
		light=mulMatVec(invWorldView, light);

		float4* matrix=traceRefraction(rayDir, rayOrig, tris, splitPlanes, splitIndices);
		if(matrix)
		{
			//float l=mulMatVec(matrix, light).z+GOOCHY;
			//pixel[0] = l*LIGHT; // red
			//pixel[1] = l*LIGHT; // green
			//pixel[2] = l*LIGHT; // blue
			pixel[0] = (pixel[0]*(1.0f-rand.w))+((matrix[0].w+1.0f)*.5f*rand.w); // red
			pixel[1] = (pixel[1]*(1.0f-rand.w))+((matrix[1].w+1.0f)*.5f*rand.w); // green
			pixel[2] = (pixel[2]*(1.0f-rand.w))+((matrix[2].w+1.0f)*.5f*rand.w); // blue
			pixel[3] = (pixel[3]*(1.0f-rand.w))+((matrix[3].w+1.0f)*.5f*rand.w); // blue	
		}
		pixel[3] = ray[3];
	}
}

__global__ void kernelRefractSpectral(unsigned char* surfaceSrc, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {

			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);

			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif


		// get a pointer to the source ray and destination pixel at (x,y)
		float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
		float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
		float4 rayDir;
		float4 rayOrig;
		
		//make uninvolved pixels pint in lightdir
#ifdef SCANLINE_MASK				
		if(ray[3]==0.0f) {
			castRay(x, y, width, height, &rayDir, &rayOrig, projection, invWorldView);
			pixel[0] = rayDir.x*.25+.5;
			pixel[1] = rayDir.y*.25+.5;
			pixel[2] = rayDir.z*.25+.5;
			pixel[3] = rayDir.w;
#ifdef CUDA_RAYPOOL
			continue;
#endif
#ifndef CUDA_RAYPOOL
			return;
#endif
		}
#endif
		__shared__ float4 light;
		light.x=0.0f;
		light.y=0.0f;
		light.z=1.0f;
		light.w=0.0f;
		
		float4 raySrf;
		refractRayBias(x, y, width, height, ray, &rayDir, &rayOrig, &raySrf, projection, invWorldView, .9f, 1.0f);
		
		float l=dot(light,raySrf)+GOOCHY;
		pixel[0] = l*LIGHT; // red
		pixel[1] = l*LIGHT; // green
		pixel[2] = l*LIGHT; // blue
		//pixel[3] = 0.5; // alpha, set to Zero for initial AO
		
		raySrf = mulMatVec(invWorldView, raySrf);
		light=mulMatVec(invWorldView, light);
		
		float4* matrix=traceRefraction(rayDir, rayOrig, tris, splitPlanes, splitIndices);
		if(matrix)
		{
			float l=mulMatVec(matrix, light).z;
			pixel[0] = (l+GOOCHY)*LIGHT; // red
		}		

		refractRayBias(x, y, width, height, ray, &rayDir, &rayOrig, &raySrf, projection, invWorldView, .8f, 1.0f);
		matrix=0;
		matrix=traceRefraction(rayDir, rayOrig, tris, splitPlanes, splitIndices);
		if(matrix)
		{
			float l=mulMatVec(matrix, light).z;
			pixel[0] += (l+GOOCHY)*.25; // red
			pixel[1] = (l+GOOCHY)*.25; // green
		}

		refractRayBias(x, y, width, height, ray, &rayDir, &rayOrig, &raySrf, projection, invWorldView, .7f, 1.0f);
		matrix=0;
		matrix=traceRefraction(rayDir, rayOrig, tris, splitPlanes, splitIndices);
		if(matrix)
		{
			float l=mulMatVec(matrix, light).z;
			pixel[1] += (l+GOOCHY)*LIGHT; // green
		}

		refractRayBias(x, y, width, height, ray, &rayDir, &rayOrig, &raySrf, projection, invWorldView, .6f, 1.0f);
		matrix=0;
		matrix=traceRefraction(rayDir, rayOrig, tris, splitPlanes, splitIndices);
		if(matrix)
		{
			float l=mulMatVec(matrix, light).z;
			pixel[1] += (l+GOOCHY)*.25; // red
			pixel[2] = (l+GOOCHY)*.25; // green
		}

		refractRayBias(x, y, width, height, ray, &rayDir, &rayOrig, &raySrf, projection, invWorldView, .5f, 1.0f);
		matrix=0;
		matrix=traceRefraction(rayDir, rayOrig, tris, splitPlanes, splitIndices);
		if(matrix)
		{
			float l=mulMatVec(matrix, light).z;
			pixel[2] += (l+GOOCHY)*LIGHT; // blue
		}
		pixel[3] = ray[3];
	}
}

__global__ void kernelRaycastBlend(unsigned char* surfaceSrc, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {

			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);

			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
		// get a pointer to the source ray and destination pixel at (x,y)
		float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
		float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
		float4 rayDir;
		float4 rayOrig;
		castRay(x, y, width, height, &rayDir, &rayOrig, projection, invWorldView);		
		
		//make uninvolved pixels pint in lightdir
#ifdef SCANLINE_MASK
		if(ray[3]==0.0f) {
			pixel[0] = rayDir.x*.25+.5;
			pixel[1] = rayDir.y*.25+.5;
			pixel[2] = rayDir.z*.25+.5;
			pixel[3] = rayDir.w;
#ifdef CUDA_RAYPOOL
			continue;
#endif
#ifndef CUDA_RAYPOOL
			return;
#endif
		}
#endif			
		float4* matrix=traceReflection(rayDir, rayOrig, tris, splitPlanes, splitIndices);
		if(matrix) {
			pixel[0] = (pixel[0]*(1.0f-rand.w))+((matrix[0].w+1.0f)*.5f*rand.w); // red
			pixel[1] = (pixel[1]*(1.0f-rand.w))+((matrix[1].w+1.0f)*.5f*rand.w); // green
			pixel[2] = (pixel[2]*(1.0f-rand.w))+((matrix[2].w+1.0f)*.5f*rand.w); // blue
			pixel[3] = (pixel[3]*(1.0f-rand.w))+((matrix[3].w+1.0f)*.5f*rand.w); // blue	
		}	
	}
}

__global__ void kernelUnitTriangleBlend(unsigned char* surfaceDst, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {

			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);

			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
		// get a pointer to the source ray and destination pixel at (x,y)
		float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
		float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
		float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;
		
		float4 rayDir;
		float4 rayOrig;
		castRay(x, y, width, height, &rayDir, &rayOrig, projection, invWorldView);

		//make uninvolved pixels pint in lightdir
#ifdef SCANLINE_MASK		
		if(ray[3]==0.0f) {
			pixel[0] = rayDir.x*.25+.5;
			pixel[1] = rayDir.y*.25+.5;
			pixel[2] = rayDir.z*.25+.5;
			pixel[3] = rayDir.w;
#ifdef CUDA_RAYPOOL
			continue;
#endif
#ifndef CUDA_RAYPOOL
			return;
#endif
		}
#endif

		float4* matrix=traceReflection(rayDir, rayOrig, tris, splitPlanes, splitIndices);
		if(matrix) {
			pixel[0] = matrix[0].x; // red
			pixel[1] = matrix[1].x; // green
			pixel[2] = matrix[2].x; // blue
			pixel[3] = matrix[3].x; // blue

			ray[0] = matrix[0].y; // red
			ray[1] = matrix[1].y; // green
			ray[2] = matrix[2].y; // blue
			ray[3] = matrix[3].y; // blue

			ray1[0] = matrix[0].w; // red
			ray1[1] = matrix[1].w; // green
			ray1[2] = matrix[2].w; // blue
			ray1[3] = matrix[3].w; // blue

			//ray[0] = matrix[3].z+1.0f; // red
			//ray[1] = matrix[1].z; // green
			//ray[2] = matrix[2].z; // blue
			//ray[3] = 1.0f; // blue

			//ray1[0] = matrix[3].w; // red
			//ray1[1] = matrix[1].w; // green
			//ray1[2] = matrix[2].w; // blue
			//ray1[3] = 1.0f; // blue
		}	
	}
}

__global__ void kernelUnitTriangleBlend2(unsigned char* surfaceDst, unsigned char* surfaceSrc, unsigned char* surfaceSrc1,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {

			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);

			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
		// get a pointer to the source ray and destination pixel at (x,y)
		float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
		float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
		float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;
		
		float4 rayDir;
		float4 rayOrig;
		castRay(x, y, width, height, &rayDir, &rayOrig, projection, invWorldView);

		//make uninvolved pixels point in lightdir
#ifdef SCANLINE_MASK
		if(ray[3]==0.0f) {
			pixel[0] = rayDir.x*.25+.5;
			pixel[1] = rayDir.y*.25+.5;
			pixel[2] = rayDir.z*.25+.5;
			pixel[3] = rayDir.w;
#ifdef CUDA_RAYPOOL
			continue;
#endif
#ifndef CUDA_RAYPOOL
			return;
#endif
		}
#endif		
		float4* matrix=traceReflection(rayDir, rayOrig, tris, splitPlanes, splitIndices);
		if(matrix) {
			pixel[0] = matrix[0].x; // red
			pixel[1] = matrix[0].y; // green
			pixel[2] = matrix[0].z; // blue
			pixel[3] = matrix[0].w; // blue

			ray[0] = matrix[1].x; // red
			ray[1] = matrix[1].y; // green
			ray[2] = matrix[1].z; // blue
			ray[3] = matrix[1].w; // blue

			ray1[0] = matrix[3].x; // red
			ray1[1] = matrix[3].y; // green
			ray1[2] = matrix[3].z; // blue
			ray1[3] = matrix[3].w; // blue
		}	
	}
}




__global__ void kernelVoxel(unsigned char* surfaceSrc, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {

			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);

			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif


		// get a pointer to the source ray and destination pixel at (x,y)
		float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
		float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;

		//make uninvolved pixels transparent end exit

		__shared__ float4 light;
		light.x=0.5773f;
		light.y=0.5773f;
		light.z=-0.5773f;
		light.w=0.0f;

		float4 rayDir;
		float4 rayOrig;
		castRay(x, y, width, height, &rayDir, &rayOrig, projection, invWorldView);

		light=mulMatVec(invWorldView, light);

		float l=traceVoxel(rayDir, rayOrig, splitPlanes, splitIndices);
		if(l>0)
		{
			pixel[0] = l/5.0+ray[0]; // red
			pixel[1] = l/5.0+ray[1]; // green
			pixel[2] = l/5.0+ray[2]; // blue
			pixel[3] = 1.0f; // blue
		}	
		else
		{
			pixel[0] = 1.0f; // red
			pixel[1] = 1.0f; // green
			pixel[2] = 0.0f; // blue
			pixel[3] = 0.0f; // blue
		}
	}
}

__global__ void kernelAmbientOcclusionBlend(unsigned char* surfaceSrc, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
		// get a pointer to the source ray and destination pixel at (x,y)
		float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
		float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;

		float4 rayDir;
		float4 rayOrig;
		float4 raySrf;
		reflectRay(x, y, width, height, ray, &rayDir, &rayOrig, &raySrf, projection, invWorldView);
		
		rayDir.x=rand.x;
		rayDir.y=rand.y;
		rayDir.z=rand.z;
		rayDir.w=0.0f;
		rayDir = normalize3(rayDir);
		
		rayDir = mulMatVec(invWorldView, rayDir);
			
		pixel[3]*=(1.0f-rand.w);
		float shadow=traceShadow(rayDir, rayOrig, tris, splitPlanes, splitIndices);
		pixel[3]+=rand.w*shadow;
			
		pixel[0]=pixel[3];			
		pixel[1]=pixel[3];			
		pixel[2]=pixel[3];					
	}
}

__global__ void kernelAmbientOcclusionBlend2(unsigned char* surfaceSrc, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
		// get a pointer to the source ray and destination pixel at (x,y)
		float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
		float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;

		float4 rayDir;
		float4 rayOrig;
		float4 raySrf;
		reflectRay(x, y, width, height, ray, &rayDir, &rayOrig, &raySrf, projection, invWorldView);
		
		rayDir.x=rand.x;
		rayDir.y=rand.y;
		rayDir.z=rand.z;
		rayDir.w=0.0f;
		rayDir = normalize3(rayDir);
		
		rayDir = mulMatVec(invWorldView, rayDir);
			
		pixel[3]*=(1.0f-rand.w);
		float shadow=traceShadow(rayDir, rayOrig, tris, splitPlanes, splitIndices);
		pixel[3]+=rand.w*shadow;
			
		pixel[0]=pixel[3];			
		pixel[1]=pixel[3];			
		pixel[2]=pixel[3];					
	}
}

__global__ void kernelAmbientOcclusionBlend3(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* worldView;
	worldView = (float4*)(matrices+4*3);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
	__shared__ float4* invWorldViewProj;
	invWorldViewProj = (float4*)(matrices+4*11);
	__shared__ float4* worldViewLast;
	worldViewLast = (float4*)(matrices+4*15);
	__shared__ float4* invWorldViewLast;
	invWorldViewLast = (float4*)(matrices+4*21);
	__shared__ float4* invWorldViewProjLast;
	invWorldViewProjLast = (float4*)(matrices+4*23);
#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
		// get a pointer to the source ray and destination pixel at (x,y)
		float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
		float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
		float* ray1 = (float*)(surfaceSrc1 + y*pitch) + 4*x;

		float4 rayDir;
		float4 rayOrig;
		float4 raySrf;
		shadowRay(x, y, width, height, ray1, &rayDir, &rayOrig, projection, invWorldView, invWorldViewLast);

		float ray_x=(ray1[0])*width;
		float ray_y=(float)height-(ray1[1]*(float)height);
		//ray_x=max(1.0f, min((float)width-2, ray_x -.5f));
		//ray_y=max(1.0f, min((float)height-2, ray_y -.5f));
		ray_x=max(1.0f, min((float)width-2, ray_x ));
		ray_y=max(1.0f, min((float)height-2, ray_y ));

		float* before = tex2D(surfaceSrc, ray_x-.0f, ray_y-.0f, pitch);
		float* before1 = tex2D(surfaceSrc, ray_x+1.0f, ray_y+1.0f, pitch);
			
		pixel[3]*=(1.0f-rand.w);
		float shadow=traceShadow(rayDir, rayOrig, tris, splitPlanes, splitIndices);
		pixel[3]+=rand.w*shadow;
			
		pixel[0]=pixel[3];			
		pixel[1]=pixel[3];			
		pixel[2]=pixel[3];					
	}
}


__global__ void kernelAmbientOcclusionBlendCull(unsigned char* surfaceSrc, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {
			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);
			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif
		// get a pointer to the source ray and destination pixel at (x,y)
		float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
		float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;

		float4 rayDir;
		float4 rayOrig;
		float4 raySrf;
		reflectRay(x, y, width, height, ray, &rayDir, &rayOrig, &raySrf, projection, invWorldView);
		
		rayDir.x=rand.x;
		rayDir.y=rand.y;
		rayDir.z=rand.z;
		rayDir.w=0.0f;
		rayDir = normalize3(rayDir);
		
		rayDir = mulMatVec(invWorldView, rayDir);
		
		if(dot(rayDir, raySrf)<0.0f)
		{
			pixel[3]*=(1.0f-rand.w);
			float shadow=traceShadow(rayDir, rayOrig, tris, splitPlanes, splitIndices);
			pixel[3]+=rand.w*shadow;
		}
			
		pixel[0]=pixel[3];			
		pixel[1]=pixel[3];			
		pixel[2]=pixel[3];					
	}
}

__global__ void kernelBentNormalBlend(unsigned char* surfaceSrc, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {

			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);

			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{
#endif

		// get a pointer to the source ray and destination pixel at (x,y)
		float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
		float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
		float4 rayDir;
		float4 rayOrig;

		//make uninvolved pixels pint in lightdir
#ifdef SCANLINE_MASK				
		if(ray[3]==0.0f) {
			castRay(x, y, width, height, &rayDir, &rayOrig, projection, invWorldView);
			pixel[0] = rayDir.x*.25+.5;
			pixel[1] = rayDir.y*.25+.5;
			pixel[2] = rayDir.z*.25+.5;
			pixel[3] = rayDir.w;
#ifdef CUDA_RAYPOOL
			continue;
#endif
#ifndef CUDA_RAYPOOL
			return;
#endif
		}
#endif

		
		float4 raySrf;
		reflectRay(x, y, width, height, ray, &rayDir, &rayOrig, &raySrf, projection, invWorldView);
		raySrf = mulMatVec(invWorldView, raySrf);

		rayDir.x=rand.x;
		rayDir.y=rand.y;
		rayDir.z=rand.z;
		rayDir = normalize3(rayDir);

		pixel[0]-=.5;
		pixel[1]-=.5;
		pixel[2]-=.5;

		pixel[0]*=(1.0f-rand.w);
		pixel[1]*=(1.0f-rand.w);
		pixel[2]*=(1.0f-rand.w);

		float shadow=traceShadow(rayDir, rayOrig, tris, splitPlanes, splitIndices);
		pixel[0]+=rayDir.x*rand.w*shadow;
		pixel[1]+=rayDir.y*rand.w*shadow;
		pixel[2]+=rayDir.z*rand.w*shadow;

		pixel[0]+=.5;
		pixel[1]+=.5;
		pixel[2]+=.5;
		//pixel[3] = ray[3];
	}
}


__global__ void kernelBentNormalBlendCull(unsigned char* surfaceSrc, unsigned char* surfaceDst,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {

			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);

			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{ //you are not supposed to understand this...
#endif
		// get a pointer to the source ray and destination pixel at (x,y)
		float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
		float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
		float4 rayDir;
		float4 rayOrig;
		
		//make uninvolved pixels pint in lightdir
#ifdef SCANLINE_MASK			
		if(ray[3]==0.0f) {
			castRay(x, y, width, height, &rayDir, &rayOrig, projection, invWorldView);
			pixel[0] = rayDir.x*.25+.5;
			pixel[1] = rayDir.y*.25+.5;
			pixel[2] = rayDir.z*.25+.5;
			pixel[3] = rayDir.w;
#ifdef CUDA_RAYPOOL
			continue;
#endif
#ifndef CUDA_RAYPOOL
			return;
#endif
		}
#endif
		float4 raySrf;
		reflectRay(x, y, width, height, ray, &rayDir, &rayOrig, &raySrf, projection, invWorldView);

		rayDir.x=rand.x;
		rayDir.y=rand.y;
		rayDir.z=rand.z;

		raySrf = mulMatVec(invWorldView, raySrf);
		//invert ray if in culling direction :)
		if(dot(rayDir, raySrf)<0.0f) {			
			rayDir.x=-rayDir.x;
			rayDir.y=-rayDir.y;
			rayDir.z=-rayDir.z;
		}
		
		float shadow=traceShadow(rayDir, rayOrig, tris, splitPlanes, splitIndices);

		pixel[0]-=.5;
		pixel[1]-=.5;
		pixel[2]-=.5;		
		pixel[0]*=(1.0f-rand.w);
		pixel[1]*=(1.0f-rand.w);
		pixel[2]*=(1.0f-rand.w);
		pixel[0]+=rayDir.x*rand.w*shadow;
		pixel[1]+=rayDir.y*rand.w*shadow;
		pixel[2]+=rayDir.z*rand.w*shadow;
		pixel[0]+=.5;
		pixel[1]+=.5;
		pixel[2]+=.5;

		return;
	}
}

__global__ void kernelBentNormalBlendCull2(unsigned char* surfaceDst, unsigned char* surfaceDst1, unsigned char* surfaceDst2, unsigned char* surfaceDst3, unsigned char* surfaceSrc, unsigned char* surfaceSrc1, unsigned char* surfaceSrc2, unsigned char* surfaceSrc3,  int width, int height, size_t pitch, float4* tris, float4* matrices, float* splitPlanes, int* splitIndices, float4 rand)
{
	//read the needed matrices for projection
	__shared__ float4* projection;
	projection = (float4*)(matrices+4*2);
	__shared__ float4* invWorldView;
	invWorldView = (float4*)(matrices+4*9);
#ifdef CUDA_RAYPOOL
	const unsigned int globalPoolRayCount = RESOLUTION_X*RESOLUTION_Y;

	// variables shared by entire warp, place to shared memory
	__shared__ volatile unsigned int nextRayArray[BLOCKDIM_Y];
	__shared__ volatile unsigned int rayCountArray[BLOCKDIM_Y];
	volatile unsigned int& localPoolNextRay = nextRayArray[threadIdx.y];
	volatile unsigned int& localPoolRayCount = rayCountArray[threadIdx.y];
	localPoolRayCount=0;

	while (true) {
		// get rays from global to local pool
		if (localPoolRayCount==0 && threadIdx.x==0) {

			localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCKDIM_X*BLOCKDIM_Y);

			localPoolRayCount = BLOCKDIM_X*BLOCKDIM_Y; 
		}
		// get rays from local pool
		unsigned int myRayIndex = localPoolNextRay + threadIdx.x;
		if (myRayIndex >= globalPoolRayCount)
			return;
		if (threadIdx.x == 0) {
			localPoolNextRay += BLOCKDIM_X;
			localPoolRayCount -= BLOCKDIM_X; 
		}
		// init and execute, these must not exit the kernel

		// get a pointer to the source ray and destination pixel at (x,y)
		unsigned int x = myRayIndex % width;
		unsigned int y = myRayIndex / width;
		if (x >= width || y >= height) continue;
#endif
#ifndef CUDA_RAYPOOL
		unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		{ //you are not supposed to understand this...
#endif
		// get a pointer to the source ray and destination pixel at (x,y)
		float* pixel = (float*)(surfaceDst + y*pitch) + 4*x;
		float* ray = (float*)(surfaceSrc + y*pitch) + 4*x;
		float4 rayDir;
		float4 rayOrig;
		//make uninvolved pixels pint in lightdir
#ifdef SCANLINE_MASK		
		if(ray[3]==0.0f) {
			castRay(x, y, width, height, &rayDir, &rayOrig, projection, invWorldView);
			pixel[0] = rayDir.x*.25+.5;
			pixel[1] = rayDir.y*.25+.5;
			pixel[2] = rayDir.z*.25+.5;
			pixel[3] = rayDir.w;
#ifdef CUDA_RAYPOOL
			continue;
#endif
#ifndef CUDA_RAYPOOL
			return;
#endif
		}
#endif		
		float4 raySrf;
		reflectRay(x, y, width, height, ray, &rayDir, &rayOrig, &raySrf, projection, invWorldView);

		rayDir.x=rand.x;
		rayDir.y=rand.y;
		rayDir.z=rand.z;

		raySrf = mulMatVec(invWorldView, raySrf);
		//invert ray if in culling direction :)
		if(dot(rayDir, raySrf)<0.0f) {			
			rayDir.x=-rayDir.x;
			rayDir.y=-rayDir.y;
			rayDir.z=-rayDir.z;
		}
		
		float shadow=traceShadow(rayDir, rayOrig, tris, splitPlanes, splitIndices);

		pixel[0]-=.5;
		pixel[1]-=.5;
		pixel[2]-=.5;		
		pixel[0]*=(1.0f-rand.w);
		pixel[1]*=(1.0f-rand.w);
		pixel[2]*=(1.0f-rand.w);
		pixel[0]+=rayDir.x*rand.w*shadow;
		pixel[1]+=rayDir.y*rand.w*shadow;
		pixel[2]+=rayDir.z*rand.w*shadow;
		pixel[0]+=.5;
		pixel[1]+=.5;
		pixel[2]+=.5;

		return;
	}
}


// global variables
__device__ unsigned int globalPoolNextRay = 0;
__global__ void resetRayCount(void)
{
	globalPoolNextRay=0;
}
bool locked=false;
int seed=1;

extern "C" 
void cuda_bsp_raytrace(void* surfaceDst, void* surfaceDst1, void* surfaceDst2, void* surfaceDst3, void* surfaceSrc, void* surfaceSrc1, void* surfaceSrc2, void* surfaceSrc3, int width, int height, size_t pitch, void* tris, void* matrices, void* splitPlanes, void* splitIndices, int method)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3( BLOCKDIM_X, BLOCKDIM_Y ); // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3( (width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y );

	float4 blend;
	blend.x=0;
	blend.y=0;
	blend.z=0;
	blend.w=1.0f;

	float4 light;
	light.x=.51f;
	light.y=.51f;
	light.z=-.51f;
	light.w=1.0f;

	float4 random;
	int rnd=std::rand()%10000;
	random.x=((double)rnd/10000.0)-.5;
	rnd=std::rand()%10000;
	random.y=((double)rnd/10000.0)-.5;
	rnd=std::rand()%10000;
	random.z=((double)rnd/10000.0)-.5;
	random.w=1.0f/256.0f;

	if(method==0)
	{
		kernelShadeTexture<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);
	}
	else if(method==1)
	{
		kernelAmbientOcclusionBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, random);
	}
	else if(method==2)
	{
		kernelAmbientOcclusionBlendCull<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, random);
	}
	else if(method==3)
	{
		kernelRaycastBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
	}
	else if(method==4)
	{
		kernelReflectBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
	}
	if(method==5)
	{	
		kernelAmbientOcclusionBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, light);				
	}
	else if(method==6)
	{
		kernelRefractBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
	}
	else if(method==7)
	{
		kernelRefractSpectral<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);
	}
	else if(method==8)
	{
		kernelVoxel<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);
	}
	else if(method==9)
	{
		kernelBentNormalBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, random);
	}
	else if(method==10)
	{
		kernelReflectBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
	}	
	else if(method==11)
	{
		random.w=1.0/128.0f;
		kernelBentNormalBlendCull<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, random);
	}
	else if(method==12)
	{
		random.w=1.0/2048.0f;
		kernelBentNormalBlendCull<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, random);
	}
	else if(method==13)
	{
		random.w=1.0/128.0f;
		kernelBentNormalBlendCull2<<<Dg,Db>>>( (unsigned char*) surfaceDst, NULL, NULL, NULL, (unsigned char*) surfaceSrc, NULL, NULL, NULL, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, random);
	}
	else if(method==14)	
	{	
		random.w=1.0/4096.0f;

		kernelBacktrackTextureBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc1,  width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*)surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);		
		kernelBentNormalBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, random);						
	}
	else if(method==15)
	{			
		kernelBacktrackTextureBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc1, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*)surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);				
		kernelBentNormalBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, random);	
	}
	else if(method==16)
	{	
		kernelBacktrackTextureBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc1, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*)surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);		
		kernelRaycastBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);		
	}
	else if(method==17)
	{	
		kernelBacktrackTextureBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc1, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);				
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*)surfaceSrc1, (unsigned char*) surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);		
		blend.w=0.1f;
		kernelRaycastBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);		
	}
	else if(method==18)
	{		
		random.w=0.1f;

		kernelBacktrackTextureBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc1, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);		
		kernelAmbientOcclusionBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, random);				
	}	
	else if(method==19)
	{	
		kernelBacktrackFixBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);	
	}	
	else if(method==20)
	{					
		kernelBacktrackTextureBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc1, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);				
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*)surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);
		random.w=1.0/64.0f;
		kernelBentNormalBlendCull<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, random);		
	}
	else if(method==21)
	{	
		kernelBacktrackTextureBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc1, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*)surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);						
		random.w=1.0/128.0f;
		kernelBentNormalBlendCull<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, random);
	}
	else if(method==22)
	{	
		kernelBacktrackTextureBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc1, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*)surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);						
		random.w=1.0/2048.0f;
		kernelBentNormalBlendCull<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, random);
	}
	else if(method==23)
	{	
		kernelBacktrackShadowBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc1, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);				
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*)surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);
		random.w=.5;
		kernelBentNormalBlendCull<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, random);
	}
	else if(method==24)
	{		
		kernelBacktrackShadowBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);				
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*)surfaceSrc, (unsigned char*) surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);		
		blend.w=0.0f;
		kernelRaycastBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);		
	}
	else if(method==25)
	{		
		kernelStencilMapBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1, (unsigned char*) surfaceSrc2, (unsigned char*) surfaceSrc3, (unsigned char*) surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);				
		//kernelCopyTexture<<<Dg,Db>>>( (unsigned char*)surfaceSrc, (unsigned char*) surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);		
		blend.w=0.0f;
		kernelRaycastBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);		
	}
	else if(method==30)
	{	
		kernelBacktrackTextureBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc1, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);				
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*)surfaceSrc1, (unsigned char*) surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);		
		blend.w=0.1f;
		kernelRaycastBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);			
	}
	else if(method==31)
	{	
		kernelBacktrackTextureBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);				
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*)surfaceSrc1, (unsigned char*) surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);		
		blend.w=0.1f;
		kernelRaycastBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);					
	}
	else if(method==32)
	{	
		kernelBacktrackTextureBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);				
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*)surfaceSrc, (unsigned char*) surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);		
		blend.w=0.1f;
		kernelRaycastBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);					
	}
	else if(method==33)
	{	
		kernelBacktrackTextureBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);				
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*)surfaceSrc, (unsigned char*) surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);		
		blend.w=0.1f;
		//kernelRaycastBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);					
	}
	else if(method==34)
	{	
		kernelBacktrackShadowBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);				
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*)surfaceSrc, (unsigned char*) surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);		
		blend.w=0.1f;
		//kernelRaycastBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);					
	}
	else if(method==35)
	{	
		kernelBacktrackShadowBlend2<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);				
		kernelCopyTexture<<<Dg,Db>>>( (unsigned char*)surfaceSrc, (unsigned char*) surfaceDst, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices);		
		blend.w=0.1f;
		//kernelRaycastBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);					
	}
	else if(method==40)
	{	
		kernelUnitTriangleBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
	}
	else if(method==41)
	{	
		kernelUnitTriangleBlend<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
	}
	else if(method==42)
	{	
		kernelUnitTriangleBlend<<<Dg,Db>>>( (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
	}
	else if(method==43)
	{	
		kernelUnitTriangleBlend2<<<Dg,Db>>>( (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
	}
	else if(method==44)
	{	
		kernelUnitTriangleBlend2<<<Dg,Db>>>( (unsigned char*) surfaceSrc1, (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
	}
	else if(method==45)
	{	
		kernelUnitTriangleBlend2<<<Dg,Db>>>( (unsigned char*) surfaceDst, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
	}
	else if(method==50)
	{	
		kernelUnitTriangleBlend<<<Dg,Db>>>( (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3,   width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
	}
	else if(method==51)
	{	
		kernelPolygonMapBlend<<<Dg,Db>>>((unsigned char*) surfaceDst, NULL, NULL, NULL, (unsigned char*) surfaceSrc1, (unsigned char*) surfaceSrc, NULL, NULL, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==52)
	{	
		kernelPolygonMapBlend2<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==53)
	{	
		kernelPolygonMapBlend3<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==54)
	{	
		kernelPolygonMapBlend4<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==55)
	{	
		kernelPolygonMapBlend5<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==56)
	{	
		kernelPolygonMapBlend6<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==57)
	{	
		kernelPolygonMapBlend7<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==58)
	{	
		kernelPolygonMapBlend8<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==59)
	{	
		kernelPolygonMapBlend9<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);
	}
	else if(method==60)
	{	
		kernelAmbientOcclusionBlend3<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3,  width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, light);				
	}
	else if(method==61)
	{	
		kernelPolygonMapBlend10<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==62)
	{	
		kernelPolygonMapBlend11<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==63)
	{	
		kernelPolygonMapBlend12<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==64)
	{	
		kernelPolygonMapBlend13<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==65)
	{	
		kernelPolygonMapBlend14<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==66)
	{	
		kernelPolygonMapBlend15<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==67)
	{	
		kernelPolygonMapBlend16<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==68)
	{	
		blend.w=0.0f;
		kernelClearTexture<<<Dg,Db>>>((unsigned char*) surfaceDst, width, height, pitch, (float4*) matrices, blend);						
		kernelPolygonMapBlend17<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, random);						
	}
	else if(method==69)
	{			
		kernelPolygonMapBlend17<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, random);						
	}
	
	else if(method==70)
	{	
		blend.x = 0.0f;
		blend.y = 0.0f;
		kernelPolygonMapBlend3<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}
	else if(method==71)
	{	
		blend.x = 1.0f;
		blend.y = 1.0f;
		kernelPolygonMapBlend3<<<Dg,Db>>>((unsigned char*) surfaceDst, (unsigned char*) surfaceDst1, (unsigned char*) surfaceDst2, (unsigned char*) surfaceDst3, (unsigned char*) surfaceSrc, (unsigned char*) surfaceSrc1,  (unsigned char*) surfaceSrc2,  (unsigned char*) surfaceSrc3, width, height, pitch, (float4*)tris, (float4*) matrices, (float*) splitPlanes, (int*) splitIndices, blend);						
	}

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("cuda_bsp_raytrace() failed to launch error = %d\n", error);
	}
}


