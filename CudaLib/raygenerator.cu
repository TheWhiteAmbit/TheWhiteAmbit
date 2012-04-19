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
#include "vectormath.cu"

__device__ void shadowRay(float x, float y, unsigned int width, unsigned int height, float* ray, float4* rayDir, float4* rayOrig, float4* projection, float4* invWorldView, float4* invWorldViewLast)
{
		//interpret the color from the scanline process
		float depth=Z_NEAR/ray[3];
		// calculate initial eye ray
		rayOrig->x = ((((float)x*2/(float)width)-1.0f)/projection[0].x)*depth;
		rayOrig->y = (-(((float)y*2/(float)height)-1.0f)/projection[1].y)*depth;
		rayOrig->z = depth;
		rayOrig->w = 1.0f;		
		*rayOrig = mulMatVec(invWorldView, *rayOrig);

		rayDir->x = 0.0f;
		rayDir->y = 0.0f;
		rayDir->z = 0.0f;
		rayDir->w = 1.0f;
		*rayDir = mulMatVec(invWorldViewLast, *rayDir);
		rayDir->x = rayDir->x-rayOrig->x;
		rayDir->y = rayDir->y-rayOrig->y;
		rayDir->z = rayDir->z-rayOrig->z;
		rayDir->w = 0.0f;
}

__device__ void shadowRayBias(unsigned int x, unsigned int y, unsigned int width, unsigned int height, float* ray, float4* rayDir, float4* rayOrig, float4* projection, float4* invWorldView, float4* invWorldViewLast, float4* invWorldViewProjLast)
{
		//interpret the color from the scanline process
		float depth=Z_NEAR/ray[3]-Z_NEAR;
		// calculate initial eye ray
		rayOrig->x = ((((float)x*2/(float)width)-1.0)/projection[0].x)*depth;
		rayOrig->y = (-(((float)y*2/(float)height)-1.0)/projection[1].y)*depth;
		rayOrig->z = depth;
		rayOrig->w = 1.0f;
		*rayOrig = mulMatVec(invWorldView, *rayOrig);

		rayDir->x = 0.0f;
		rayDir->y = 0.0f;
		rayDir->z = 0.0f;
		rayDir->w = 1.0f;
		*rayDir = mulMatVec(invWorldViewLast, *rayDir);
		rayDir->x = rayDir->x-rayOrig->x;
		rayDir->y = rayDir->y-rayOrig->y;
		rayDir->z = rayDir->z-rayOrig->z;
		rayDir->w = 0.0f;
}

__device__ void reflectRay(unsigned int x, unsigned int y, unsigned int width, unsigned int height, float* ray, float4* rayDir, float4* rayOrig, float4* raySrf, float4* projection, float4* invWorldView)
{
		//interpret the color from the scanline process
		raySrf->x=ray[0]*2.0f-1.0f;
		raySrf->y=ray[1]*2.0f-1.0f;
		raySrf->z=ray[2]*2.0f-1.0f;
		raySrf->w=Z_NEAR/ray[3];
		// calculate initial eye ray
		float4 rayCam;
		rayCam.x = (((float)x*2/(float)width)-1.0f)/projection[0].x;
		rayCam.y = -(((float)y*2/(float)height)-1.0f)/projection[1].y;
		rayCam.z = 1.0f;
		rayCam.w = 0.0f;
		// calculate depth position in screen space
		rayOrig->x = rayCam.x*raySrf->w;
		rayOrig->y = rayCam.y*raySrf->w;
		rayOrig->z = rayCam.z*raySrf->w;
		rayOrig->w = 1.0f;
		// calculate reflective ray direction
		float lengthSq2=2.0f*(raySrf->x*rayCam.x+raySrf->y*rayCam.y+raySrf->z*rayCam.z);
		rayDir->x = rayCam.x-lengthSq2*raySrf->x;
		rayDir->y = rayCam.y-lengthSq2*raySrf->y;
		rayDir->z = rayCam.z-lengthSq2*raySrf->z;
		rayDir->w = 0.0f;
		normalize3(rayDir); //TODO: important look at other places

		*rayDir = mulMatVec(invWorldView, *rayDir);
		*rayOrig = mulMatVec(invWorldView, *rayOrig);
		raySrf->w=0.0f;
}

__device__ void reflectRayBias(unsigned int x, unsigned int y, unsigned int width, unsigned int height, float* ray, float4* rayDir, float4* rayOrig, float4* raySrf, float4* projection, float4* invWorldView)
{
		//interpret the color from the scanline process
		raySrf->x=ray[0]*2.0f-1.0f;
		raySrf->y=ray[1]*2.0f-1.0f;
		raySrf->z=ray[2]*2.0f-1.0f;
		raySrf->w=Z_NEAR/ray[3]-Z_NEAR;
		// calculate initial eye ray
		float4 rayCam;
		rayCam.x = (((float)x*2/(float)width)-1.0f)/projection[0].x;
		rayCam.y = -(((float)y*2/(float)height)-1.0f)/projection[1].y;
		rayCam.z = 1.0f;
		rayCam.w = 0.0f;
		// calculate depth position in screen space
		rayOrig->x = rayCam.x*raySrf->w;
		rayOrig->y = rayCam.y*raySrf->w;
		rayOrig->z = rayCam.z*raySrf->w;
		rayOrig->w = 1.0f;
		// calculate reflective ray direction
		float lengthSq2=2.0f*(raySrf->x*rayCam.x+raySrf->y*rayCam.y+raySrf->z*rayCam.z);
		rayDir->x = rayCam.x-lengthSq2*raySrf->x;
		rayDir->y = rayCam.y-lengthSq2*raySrf->y;
		rayDir->z = rayCam.z-lengthSq2*raySrf->z;
		rayDir->w = 0.0f;
		normalize3(rayDir); //TODO: important look at other places
		
		*rayDir = mulMatVec(invWorldView, *rayDir);
		*rayOrig = mulMatVec(invWorldView, *rayOrig);
		raySrf->w=0.0f;
}

__device__ void refractRay(unsigned int x, unsigned int y, unsigned int width, unsigned int height, float* ray, float4* rayDir, float4* rayOrig, float4* raySrf, float4* projection, float4* invWorldView, float levelCamera, float levelSurface)
{
		//interpret the color from the scanline process
		raySrf->x=ray[0]*2.0f-1.0f;
		raySrf->y=ray[1]*2.0f-1.0f;
		raySrf->z=ray[2]*2.0f-1.0f;
		raySrf->w=Z_NEAR/ray[3];
		// calculate initial eye ray
		float4 rayCam;
		rayCam.x = (((float)x*2/(float)width)-1.0f)/projection[0].x;
		rayCam.y = -(((float)y*2/(float)height)-1.0f)/projection[1].y;
		rayCam.z = 1.0f;
		rayCam.w = 0.0f;
		// calculate depth position in screen space
		rayOrig->x = rayCam.x*raySrf->w;
		rayOrig->y = rayCam.y*raySrf->w;
		rayOrig->z = rayCam.z*raySrf->w;
		rayOrig->w = 1.0f;
		// calculate reflective ray direction
		float lengthSq=(raySrf->x*rayCam.x+raySrf->y*rayCam.y+raySrf->z*rayCam.z);
		rayDir->x = rayCam.x+levelCamera*lengthSq*raySrf->x*levelSurface;
		rayDir->y = rayCam.y+levelCamera*lengthSq*raySrf->y*levelSurface;
		rayDir->z = rayCam.z+levelCamera*lengthSq*raySrf->z*levelSurface;
		rayDir->w = 0.0f;
		normalize3(rayDir); //TODO: important look at other places

		*rayDir = mulMatVec(invWorldView, *rayDir);
		*rayOrig = mulMatVec(invWorldView, *rayOrig);
		raySrf->w=0.0f;
}


__device__ void refractRayBias(unsigned int x, unsigned int y, unsigned int width, unsigned int height, float* ray, float4* rayDir, float4* rayOrig, float4* raySrf, float4* projection, float4* invWorldView, float levelCamera, float levelSurface)
{
		//interpret the color from the scanline process
		raySrf->x=ray[0]*2.0f-1.0f;
		raySrf->y=ray[1]*2.0f-1.0f;
		raySrf->z=ray[2]*2.0f-1.0f;
		raySrf->w=Z_NEAR/ray[3]+Z_NEAR;;
		// calculate initial eye ray
		float4 rayCam;
		rayCam.x = (((float)x*2/(float)width)-1.0f)/projection[0].x;
		rayCam.y = -(((float)y*2/(float)height)-1.0f)/projection[1].y;
		rayCam.z = 1.0f;
		rayCam.w = 0.0f;
		// calculate depth position in screen space
		rayOrig->x = rayCam.x*raySrf->w;
		rayOrig->y = rayCam.y*raySrf->w;
		rayOrig->z = rayCam.z*raySrf->w;
		rayOrig->w = 1.0f;
		// calculate reflective ray direction
		float lengthSq=(raySrf->x*rayCam.x+raySrf->y*rayCam.y+raySrf->z*rayCam.z);
		rayDir->x = rayCam.x+levelCamera*lengthSq*raySrf->x*levelSurface;
		rayDir->y = rayCam.y+levelCamera*lengthSq*raySrf->y*levelSurface;
		rayDir->z = rayCam.z+levelCamera*lengthSq*raySrf->z*levelSurface;
		rayDir->w = 0.0f;
		normalize3(rayDir); //TODO: important look at other places

		*rayDir = mulMatVec(invWorldView, *rayDir);
		*rayOrig = mulMatVec(invWorldView, *rayOrig);
		raySrf->w=0.0f;
}

__device__ void castRay(unsigned int x, unsigned int y, unsigned int width, unsigned int height, float4* rayDir, float4* rayOrig, float4* projection, float4* invWorldView)
{	
		// calculate initial eye ray
		rayDir->x = (((float)x*2/(float)width)-1.0f)/projection[0].x;
		rayDir->y = -(((float)y*2/(float)height)-1.0f)/projection[1].y;
		rayDir->z = 1.0f;
		rayDir->w = 0.0f;
		normalize3(rayDir); //TODO: important look at other places

		rayOrig->x = 0.0;
		rayOrig->y = 0.0;
		rayOrig->z = 0.0;
		rayOrig->w = 1.0f;

		*rayDir = mulMatVec(invWorldView, *rayDir);
		*rayOrig = mulMatVec(invWorldView, *rayOrig);
}
