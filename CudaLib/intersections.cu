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


__device__ float intersectTriangle(float4* matrix, float4 rayDir, float4 rayOrig)
{	
	float4 o=mulMatVec(matrix, rayOrig);
	float4 d=mulMatVec(matrix, rayDir);

	float t=-o.z/d.z;
	float u=o.x+t*d.x;
	float v=o.y+t*d.y;

	return min(u, min(v, min(1.0f-u-v, t)));
}

__device__ float intersectTriangleZNear(float4* matrix, float4 rayDir, float4 rayOrig, float zNear)
{	
	float4 o=mulMatVec(matrix, rayOrig);
	float4 d=mulMatVec(matrix, rayDir);

	float t=-o.z/d.z;
	float u=o.x+t*d.x;
	float v=o.y+t*d.y;

	return min(u, min(v, min(1.0f-u-v, t-zNear)));
}

__device__ float intersectTriangle(float4* matrix, float4 rayDir, float4 rayOrig, float* t)
{	
	float4 o=mulMatVec(matrix, rayOrig);
	float4 d=mulMatVec(matrix, rayDir);

	*t=-o.z/d.z;
	float u=o.x+*t*d.x;
	float v=o.y+*t*d.y;

	return min(u, min(v, min(1.0f-u-v, *t)));
}

__device__ float intersectTriangleZNear(float4* matrix, float4 rayDir, float4 rayOrig, float* t, float zNear)
{	
	float4 o=mulMatVec(matrix, rayOrig);
	float4 d=mulMatVec(matrix, rayDir);

	*t=-o.z/d.z;
	float u=o.x+*t*d.x;
	float v=o.y+*t*d.y;

	return min(u, min(v, min(1.0f-u-v, *t-zNear)));
}

__device__ void intersectTriangleZOnly(float4* matrix, float4 rayDir, float4 rayOrig, float* t, float zNear)
{	
	*t=-mulMatVecZOnly(matrix, rayOrig)/mulMatVecZOnly(matrix, rayDir);
}

__device__ float intersectTriangleZFar(float4* matrix, float4 rayDir, float4 rayOrig, float zFar, float* t)
{	
	float4 o=mulMatVec(matrix, rayOrig);
	float4 d=mulMatVec(matrix, rayDir);

	*t=-o.z/d.z;
	float u=o.x+*t*d.x;
	float v=o.y+*t*d.y;

	return min(u, min(v, min(1.0f-u-v, min(zFar-*t, *t))));
}

__device__ float intersectTriangleZFarFront(float4* matrix, float4 rayDir, float4 rayOrig, float zFar, float* t)
{	
	float4 o=mulMatVec(matrix, rayOrig);
	float4 d=mulMatVec(matrix, rayDir);

	*t=-o.z/d.z;
	float u=o.x+*t*d.x;
	float v=o.y+*t*d.y;

	return min(u, min(v, min(1.0f-u-v, min(zFar-*t,min( d.z, *t)))));
}

__device__ float intersectTriangleZNearZFar(float4* matrix, float4 rayDir, float4 rayOrig, float zNear, float zFar, float* t)
{	
	float4 o=mulMatVec(matrix, rayOrig);
	float4 d=mulMatVec(matrix, rayDir);

	*t=-o.z/d.z;
	float u=o.x+*t*d.x;
	float v=o.y+*t*d.y;

	return min(u, min(v, min(1.0f-u-v, min(zFar-*t, *t-zNear))));
}

__device__ float intersectTriangleFront(float4* matrix, float4 rayDir, float4 rayOrig, float zNear, float* t)
{	
	float4 o=mulMatVec(matrix, rayOrig);
	float4 d=mulMatVec(matrix, rayDir);

	*t=-o.z/d.z;
	float u=o.x+*t*d.x;
	float v=o.y+*t*d.y;

	return min(u, min(v, min(1.0f-u-v, min(zNear-*t, min( d.z, *t)))));
	//return min(u, min(v, min(1.0f-u-v, *t)));
}

__device__ float intersectTriangleBack(float4* matrix, float4 rayDir, float4 rayOrig, float zNear, float* t)
{	
	float4 o=mulMatVec(matrix, rayOrig);
	float4 d=mulMatVec(matrix, rayDir);

	*t=-o.z/d.z;
	float u=o.x+*t*d.x;
	float v=o.y+*t*d.y;

	return min(u, min(v, min(1.0f-u-v, min(zNear-*t, min(-d.z, *t)))));
	//return min(u, min(v, min(1.0f-u-v, *t)));
}

__device__ float intersectDiscZFar(float4* matrix, float4 rayDir, float4 rayOrig, float zFar, float* t)
{	
	float4 o=mulMatVec(matrix, rayOrig);
	float4 d=mulMatVec(matrix, rayDir);

	*t=-o.z/d.z;
    float u=o.x+*t*d.x-.25;
    float v=o.y+*t*d.y-.25;
	//float w=o.z+*t*d.z;

	return min(1.0f-(u*u+v*v)*8.0, min(zFar-*t, *t));
}

//TODO: This isn`t working :(
__device__ float intersectSphereZFar(float4* matrix, float4 rayDir, float4 rayOrig, float zFar, float* t)
{	
	float4 o=mulMatVec(matrix, rayOrig);
	float4 d=mulMatVec(matrix, rayDir);

	float b=dot(o, d);
	float c=dot(d ,d) - 0.00001f;
	float result=b*b-c;
	//return result > 0 ? -b - sqrt(result) : FLOAT_NEG_INF;
	return result;
}