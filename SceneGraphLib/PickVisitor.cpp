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



#include "PickVisitor.h"
#include "Mesh.h"
namespace TheWhiteAmbit {
	PickVisitor::PickVisitor(HWND a_WindowHandle)
	{
		m_ScreenPick=PickScreen(a_WindowHandle);
		m_pWorldViewRay=NULL;
	}

	PickVisitor::PickVisitor(D3DXVECTOR2 a_ClickPoint)
	{	
		m_ScreenPick=a_ClickPoint;
		m_pWorldViewRay=NULL;
	}

	PickVisitor::PickVisitor(Ray a_Ray)
	{	
		m_pWorldViewRay=&a_Ray;
		m_pWorldViewRay->dir=a_Ray.dir;
		m_pWorldViewRay->orig=a_Ray.orig;
	}

	PickVisitor::~PickVisitor(void)
	{
	}

	void PickVisitor::visit(IPickable* a_pSceneNode)
	{
		if(m_Intersection.count==0)
		{
			Ray worldViewRay;
			if(!m_pWorldViewRay)
			{
				worldViewRay.orig = D3DXVECTOR4(0,0,0,1);	//the camera in worldView ist always where the camera is ;)
				worldViewRay.dir = this->PickDirection(m_ScreenPick, a_pSceneNode->getProjection());
			}
			else worldViewRay = *m_pWorldViewRay;

			Ray objectSpaceRay = this->PickInvMatrix(worldViewRay, a_pSceneNode->getWorld() * a_pSceneNode->getView());

			double dist=a_pSceneNode->intersect(objectSpaceRay);
			if(dist>=0.0)
			{
				m_Intersection.count=1;
				m_Intersection.dwFace=222;
				m_Intersection.fBary1=0.9;
				m_Intersection.fBary2=0.8;
				m_Intersection.fDist=(FLOAT)dist;
				m_Intersection.tu=.5;
				m_Intersection.tv=.5;
				m_Intersection.pMesh=NULL;
			}
		}
	}

	Intersection PickVisitor::getIntersection(void)
	{
		return m_Intersection;
	}

	D3DXVECTOR2 PickVisitor::PickScreen(HWND hWND)
	{
		// Get the Pick ray from the mouse position
		POINT cursor;
		RECT rect;

		GetCursorPos( &cursor );
		ScreenToClient( hWND, &cursor );
		GetClientRect( hWND, &rect);

		D3DXVECTOR2 result;
		result.x = ( ( ( 2.0f * cursor.x) / rect.right-rect.left ) - 1 );
		result.y = -( ( ( 2.0f * cursor.y) / rect.bottom-rect.top ) - 1 );
		return result;
	}

	D3DXVECTOR4 PickVisitor::PickDirection(
		D3DXVECTOR2 a_ScreenPick, 
		D3DXMATRIX matProj)
	{
		// Compute the vector of the Pick ray in screen space
		D3DXVECTOR4 result;
		result.x = a_ScreenPick.x / matProj._11;
		result.y = a_ScreenPick.y / matProj._22;
		result.z = 1.0f;
		result.w = 0.0f;
		return result;
	}


	Ray PickVisitor::PickInvMatrix(
		Ray ray, 
		D3DXMATRIX a_Matrix)
	{
		// Get the inverse view matrix

		D3DXMATRIX matrix;
		D3DXMatrixInverse( &matrix, NULL, &a_Matrix );

		Ray result;
		// Transform the pick ray (normaly WorldView) into object space
		D3DXVec4Transform(&result.orig, &ray.orig, &matrix);
		D3DXVec4Transform(&result.dir, &ray.dir, &matrix);

		//TODO: test ray with and without normalize
		D3DXVec3Normalize((D3DXVECTOR3*)&result.dir, (D3DXVECTOR3*)&result.dir);
		result.dir.w=0.0;
		return result;
	}


	////--------------------------------------------------------------------------------------
	//// Given a ray origin (orig) and direction (dir), and three vertices of a triangle, this
	//// function returns TRUE and the interpolated texture coordinates if the ray intersects 
	//// the triangle
	////--------------------------------------------------------------------------------------
	//bool PickVisitor::IntersectRayFace( const Ray& ray, const Face& tri,
	//                        FLOAT* t, FLOAT* u, FLOAT* v , BOOL* backface)
	//{
	//    // Find vectors for two edges sharing vert0
	//    D3DXVECTOR3 edge1 = tri.v1.pos - tri.v0.pos;
	//    D3DXVECTOR3 edge2 = tri.v2.pos - tri.v0.pos;
	//
	//    // Begin calculating determinant - also used to calculate U parameter
	//    D3DXVECTOR3 pvec;
	//    D3DXVec3Cross( &pvec, &ray.dir, &edge2 );
	//
	//    // If determinant is near zero, ray lies in plane of triangle
	//    FLOAT det = D3DXVec3Dot( &edge1, &pvec );
	//
	//    D3DXVECTOR3 tvec;
	//    if( det > 0 )
	//    {
	//		tvec = ray.orig - tri.v0.pos;
	//		*backface=false;
	//    }
	//    else
	//    {
	//		tvec = tri.v0.pos - ray.orig;
	//        det = -det;
	//		*backface=true;
	//    }
	//
	//    if( det < 0.0001f )
	//        return FALSE;
	//
	//    // Calculate U parameter and test bounds
	//    *u = D3DXVec3Dot( &tvec, &pvec );
	//    if( *u < 0.0f || *u > det )
	//        return FALSE;
	//
	//    // Prepare to test V parameter
	//    D3DXVECTOR3 qvec;
	//    D3DXVec3Cross( &qvec, &tvec, &edge1 );
	//
	//    // Calculate V parameter and test bounds
	//    *v = D3DXVec3Dot( &ray.dir, &qvec );
	//    if( *v < 0.0f || *u + *v > det )
	//        return FALSE;
	//
	//    // Calculate t, scale parameters, ray intersects triangle
	//    *t = D3DXVec3Dot( &edge2, &qvec );
	//    FLOAT fInvDet = 1.0f / det;
	//    *t *= fInvDet;
	//    *u *= fInvDet;
	//    *v *= fInvDet;
	//
	//    return TRUE;
	//}
}