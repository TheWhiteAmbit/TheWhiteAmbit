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



#ifndef CUDA_VECTORMATH
#define CUDA_VECTORMATH

#include <stdio.h>
#include <stdlib.h>

#define PI				3.1415926536f
#define FLOAT_INF		0x7f800000
#define FLOAT_NEG_INF	0xff800000

__device__ float dot(float* v0, float4 v1)
{
	return v0[0]*v1.x+v0[1]*v1.y+v0[2]*v1.z+v0[3]*v1.w;
}

__device__ float dot(float4 v0, float4 v1)
{
	return v0.x*v1.x+v0.y*v1.y+v0.z*v1.z+v0.w*v1.w;
}

__device__ float4 mulMatVec(float4* mat, float4 vec)
{
	float4 result;
	result.x = vec.x*mat[0].x+vec.y*mat[1].x+vec.z*mat[2].x+vec.w*mat[3].x;
	result.y = vec.x*mat[0].y+vec.y*mat[1].y+vec.z*mat[2].y+vec.w*mat[3].y;
	result.z = vec.x*mat[0].z+vec.y*mat[1].z+vec.z*mat[2].z+vec.w*mat[3].z;
	result.w = vec.x*mat[0].w+vec.y*mat[1].w+vec.z*mat[2].w+vec.w*mat[3].w;
	return result;
}

__device__ float mulMatVecZOnly(float4* mat, float4 vec)
{
	return vec.x*mat[0].z+vec.y*mat[1].z+vec.z*mat[2].z+vec.w*mat[3].z;
}

__device__ float4 normalize3(float4 vec)
{
	float length=sqrt(vec.x*vec.x+vec.y*vec.y+vec.z*vec.z);
	float4 result;
	result.x = vec.x/length;
	result.y = vec.y/length;
	result.z = vec.z/length;
	result.w = vec.w;
	return result;
}

__device__ void normalize3(float4* vec)
{
	float length=sqrt(vec->x*vec->x+vec->y*vec->y+vec->z*vec->z);	
	vec->x = vec->x/length;
	vec->y = vec->y/length;
	vec->z = vec->z/length;
	vec->w = vec->w;	
}

__device__ float* restorePlane(float* vec)
{
	if(vec[3]<1.0)
		return vec;
	float length=sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);	
	float4 result;
	result.w = vec[3]/length;

	result.x = vec[0] * result.w;
	result.y = vec[1] * result.w;
	result.z = vec[2] * result.w;

	vec[0] = result.x;
	vec[1] = result.y;
	vec[2] = result.z;
	vec[3] = result.w;
	return vec;
}

// lerp
__device__ float4 lerp4(float4 a, float4 b, float t)
{
	float4 result;
	result.x = a.x + t*(b.x-a.x);
	result.y = a.y + t*(b.y-a.y);
	result.z = a.z + t*(b.z-a.z);
	result.w = a.w + t*(b.w-a.w);
	return result;
}

// lerp
__device__ float4 lerp4(float* a, float* b, float t)
{
	float4 result;
 	result.x = a[0] + t*(b[0]-a[0]);
	result.y = a[1] + t*(b[1]-a[1]);
	result.z = a[2] + t*(b[2]-a[2]);
	result.w = a[3] + t*(b[3]-a[3]);
	return result;
}

__device__ float* tex2D(unsigned char* surface, int px, int py, int pitch){
	return ((float*)(surface + py * pitch) + 4*px);
}

__device__ float4* tex2DTriangleMatrix(unsigned char* surface, unsigned char* surface1, unsigned char* surface2, float x, float y, size_t pitch)
{
	float* before1 = tex2D(surface, x, y, pitch);
	float* before2 = tex2D(surface1, x, y, pitch);
	float* before3 = tex2D(surface2, x, y, pitch);

	float4 triangle[4];
	triangle[0].x = before1[0];
	triangle[1].x = before1[1];
	triangle[2].x = before1[2];
	triangle[3].x = before1[3];

	triangle[0].y = before2[0];
	triangle[1].y = before2[1];
	triangle[2].y = before2[2];
	triangle[3].y = before2[3];

	triangle[0].z = before3[0];
	triangle[1].z = before3[1];
	triangle[2].z = before3[2];
	triangle[3].z = before3[3]-1.0;

	triangle[0].w = before3[0];
	triangle[1].w = before3[1];
	triangle[2].w = before3[2];
	triangle[3].w = before3[3];

	return &triangle[0];
}

__device__ float4* tex2DTriangleMatrixZOnly(unsigned char* surface, unsigned char* surface1, unsigned char* surface2, float x, float y, size_t pitch)
{
	float* before3 = tex2D(surface2, x, y, pitch);

	float4 triangle[4];

	triangle[0].z = before3[0];
	triangle[1].z = before3[1];
	triangle[2].z = before3[2];
	triangle[3].z = before3[3]-1.0;

	return &triangle[0];
}

__device__ float4 tex2DBilinear(unsigned char* surface, float x, float y, size_t pitch)
{
    float px = floorf(x);   // integer position
    float py = floorf(y);
    float fx = x - px;      // fractional position
    float fy = y - py;
	int pxi=px;
	int pyi=py;
    return lerp4( lerp4( tex2D(surface, pxi, pyi, pitch),        tex2D(surface, pxi + 1, pyi, pitch), fx ),
                 lerp4( tex2D(surface, pxi, pyi + 1, pitch), tex2D(surface, pxi + 1.0f, pyi + 1, pitch), fx ), fy );
}

__device__ float4 tex2DPlanesIntersections(unsigned char* surface, float x, float y, size_t pitch, float4 rayOrig, float4 rayDir)
{
	//TODO: implement Plane intersections on each pixel, give each hitdistance in one vector component
    float px = floorf(x);   // integer position
    float py = floorf(y);
    float fx = x - px;      // fractional position
    float fy = y - py;
	int pxi=px;
	int pyi=py;

	float* intersect0=restorePlane(tex2D(surface, pxi, pyi, pitch));
	float* intersect1=restorePlane(tex2D(surface, pxi+1, pyi, pitch));
	float* intersect2=restorePlane(tex2D(surface, pxi, pyi+1, pitch));
	float* intersect3=restorePlane(tex2D(surface, pxi+1, pyi+1, pitch));

	float orig0=dot(intersect0, rayOrig);
	float dir0=dot(intersect0, rayDir);
	float orig1=dot(intersect1, rayOrig);
	float dir1=dot(intersect1, rayDir);
	float orig2=dot(intersect2, rayOrig);
	float dir2=dot(intersect2, rayDir);
	float orig3=dot(intersect3, rayOrig);
	float dir3=dot(intersect3, rayDir);
	
	float4 result;
	result.x = -orig0/dir0;
	result.y = -orig1/dir1;
	result.z = -orig2/dir2;
	result.w = -orig3/dir3;
	return result;		
}

#endif