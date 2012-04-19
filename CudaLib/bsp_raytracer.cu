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
#include "raygenerator.cu"
#include "intersections.cu"


__device__ float traceShadow(float4 rayDir, float4 rayOrig, float4* tris, float* splitPlanes, int* splitIndices)
{
	//create stack
	int stackIndex;
	int splitStack[BSP_MAX_DEPTH];
	float inStack[BSP_MAX_DEPTH];
	float outStack[BSP_MAX_DEPTH];

	//init stack
	stackIndex=0;
	splitStack[0]=0;
	inStack[0]=Z_NEAR;
	outStack[0]=FLOAT_INF;

	while( stackIndex>=0 )
	{	
		//pop stack
		float t_in=inStack[stackIndex];
		float t_out=outStack[stackIndex];
		int currentNode=splitStack[stackIndex];
		stackIndex--;

		while( !splitIndices[currentNode] )
		{
			float* plane=(float*)(splitPlanes+currentNode);
			float orig=dot(plane, rayOrig);
			float dir=dot(plane, rayDir);
			float t_split=-orig/dir;

			int first;
			int second;
			if( dir>=0.0 ){
				first=splitIndices[currentNode+1];
				second=splitIndices[currentNode+2];
			}else{
				second=splitIndices[currentNode+1];
				first=splitIndices[currentNode+2];
			}

			if( first && t_split>t_in ){
				if( second && t_split<t_out){
					//push stack
					stackIndex++;
					splitStack[stackIndex]=second;
					inStack[stackIndex]=max(t_split, t_in);
					outStack[stackIndex]=t_out;
				}
				currentNode=first;
				t_out=min(t_split, t_out);
			}
			else if( second && t_split<t_out ){
				currentNode=second;
				t_in=max(t_split, t_in);
			}
			else if(stackIndex>0){
				//pop stack
				t_in=inStack[stackIndex];
				t_out=outStack[stackIndex];
				currentNode=splitStack[stackIndex];
				stackIndex--;
			}
			else break;
		}

		float4* matrix=(float4*)(tris+splitIndices[currentNode+3]);
		for( int iTri=0; iTri<splitIndices[currentNode]; iTri++ )
		{
			if(intersectTriangleZNear(matrix, rayDir, rayOrig, 0.0005f ) >= 0.0f){
			//if(intersectTriangleZFar(matrix, rayDir, rayOrig, FLOAT_INF, &t) >= 0.0f){
					return 0.0f;
			}
			matrix += 4;
		}
	}	
	return 1.0f;
}


__device__ float4* traceReflection(float4 rayDir, float4 rayOrig, float4* tris, float* splitPlanes, int* splitIndices)
{
	float4* result=0;
	//create stack
	int stackIndex;
	int splitStack[BSP_MAX_DEPTH];
	float inStack[BSP_MAX_DEPTH];
	float outStack[BSP_MAX_DEPTH];

	//init stack
	stackIndex=0;
	splitStack[0]=0;
	inStack[0]=Z_NEAR;
	outStack[0]=FLOAT_INF;

	float nearestHit=FLOAT_INF;
	while( stackIndex>=0 )
	{	
		//pop stack
		float t_in=inStack[stackIndex];
		float t_out=outStack[stackIndex];
		int currentNode=splitStack[stackIndex];
		stackIndex--;

		while( !splitIndices[currentNode] )
		{
			float* plane=(float*)(splitPlanes+currentNode);
			float orig=dot(plane, rayOrig);
			float dir=dot(plane, rayDir);
			float t_split=-orig/dir;

			int first;
			int second;
			if( dir>=0.0 ){
				first=splitIndices[currentNode+1];
				second=splitIndices[currentNode+2];
			}else{
				second=splitIndices[currentNode+1];
				first=splitIndices[currentNode+2];
			}

			if( first && t_split > t_in ){
				if( second && t_split < t_out){
					//push stack
					stackIndex++;
					splitStack[stackIndex]=second;
					inStack[stackIndex]=max(t_split, t_in);
					outStack[stackIndex]=t_out;
				}
				currentNode=first;
				t_out=min(t_split, t_out);
			}
			else if( second && t_split < t_out ){
				currentNode=second;
				t_in=max(t_split, t_in);
			}
			else if(stackIndex > 0){
				//pop stack
				t_in=inStack[stackIndex];
				t_out=outStack[stackIndex];
				currentNode=splitStack[stackIndex];
				stackIndex--;
			}
			else break;
		}

		float4* matrix=(float4*)(tris+splitIndices[currentNode+3]);
		for( int iTri=0; iTri < splitIndices[currentNode]; iTri++ )
		{
			float t;
			if(intersectTriangleZFar(matrix, rayDir, rayOrig, nearestHit, &t) >= 0.0f){
					nearestHit=t;
					result=matrix;
			}
			matrix += 4;
		}
		if( nearestHit <= t_out )
			break;
	}	
	return result;
}


__device__ float4* traceRefraction(float4 rayDir, float4 rayOrig, float4* tris, float* splitPlanes, int* splitIndices)
{
	float4* result=0;
	//create stack
	int stackIndex;
	int splitStack[BSP_MAX_DEPTH];
	float inStack[BSP_MAX_DEPTH];
	float outStack[BSP_MAX_DEPTH];

	//init stack
	stackIndex=0;
	splitStack[0]=0;
	inStack[0]=Z_NEAR;
	outStack[0]=FLOAT_INF;

	float nearestHit=FLOAT_INF;
	while( stackIndex>=0 )
	{	
		//pop stack
		float t_in=inStack[stackIndex];
		float t_out=outStack[stackIndex];
		int currentNode=splitStack[stackIndex];
		stackIndex--;

		while( !splitIndices[currentNode] )
		{
			float* plane=(float*)(splitPlanes+currentNode);
			float orig=dot(plane, rayOrig);
			float dir=dot(plane, rayDir);
			float t_split=-orig/dir;

			int first;
			int second;
			if( dir>=0.0 ){
				first=splitIndices[currentNode+1];
				second=splitIndices[currentNode+2];
			}else{
				second=splitIndices[currentNode+1];
				first=splitIndices[currentNode+2];
			}

			if( first && t_split>=t_in ){
				if( second && t_split<t_out){
					//push stack
					stackIndex++;
					splitStack[stackIndex]=second;
					inStack[stackIndex]=max(t_split, t_in);
					outStack[stackIndex]=t_out;
				}
				currentNode=first;
				t_out=min(t_split, t_out);
			}
			else if( second && t_split<=t_out ){
				currentNode=second;
				t_in=max(t_split, t_in);
			}
			else if(stackIndex>0){
				//pop stack
				t_in=inStack[stackIndex];
				t_out=outStack[stackIndex];
				currentNode=splitStack[stackIndex];
				stackIndex--;
			}
			else break;
		}

		float4* matrix=(float4*)(tris+splitIndices[currentNode+3]);
		for( int iTri=0; iTri<splitIndices[currentNode]; iTri++ )
		{
			float t;
			if(intersectTriangleZNearZFar(matrix, rayDir, rayOrig, Z_NEAR, nearestHit, &t) >= 0.0f){
			//if(intersectTriangleBack(matrix, rayDir, rayOrig, nearestHit, &t) >= 0.0f){
					nearestHit=t;
					result=matrix;
			}
			matrix += 4;
		}
		if( nearestHit<=t_out )
			break;
	}	
	return result;
}


__device__ float traceVoxel(float4 rayDir, float4 rayOrig, float* splitPlanes, int* splitIndices)
{
	//create stack
	int stackIndex;
	int splitStack[BSP_MAX_DEPTH];
	float inStack[BSP_MAX_DEPTH];
	float outStack[BSP_MAX_DEPTH];

	//init stack
	stackIndex=0;
	splitStack[0]=0;
	inStack[0]=Z_NEAR;
	outStack[0]=FLOAT_INF;

	while( stackIndex>=0 )
	{	
		//pop stack
		float t_in=inStack[stackIndex];
		float t_out=outStack[stackIndex];
		int currentNode=splitStack[stackIndex];
		stackIndex--;
		float t_split;
		while( !splitIndices[currentNode] )
		{
			float* plane=(float*)(splitPlanes+currentNode);
			float orig=dot(plane, rayOrig);
			float dir=dot(plane, rayDir);
			t_split=-orig/dir;

			int first;
			int second;
			if( dir>=0.0 ){
				first=splitIndices[currentNode+1];
				second=splitIndices[currentNode+2];
			}else{
				second=splitIndices[currentNode+1];
				first=splitIndices[currentNode+2];
			}

			if( first && t_split>t_in ){
				if( second && t_split<t_out){
					//push stack
					stackIndex++;
					splitStack[stackIndex]=second;
					inStack[stackIndex]=max(t_split, t_in);
					outStack[stackIndex]=t_out;
				}
				currentNode=first;
				t_out=min(t_split, t_out);
			}
			else if( second && t_split<t_out ){
				currentNode=second;
				t_in=max(t_split, t_in);
			}
			else if(stackIndex>0){
				//pop stack
				t_in=inStack[stackIndex];
				t_out=outStack[stackIndex];
				currentNode=splitStack[stackIndex];
				stackIndex--;
			}
			else break;
		}
		if(splitIndices[currentNode])
			return (float)t_split;
	}	
	return -1;
}
