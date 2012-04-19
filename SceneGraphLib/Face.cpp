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




#include "Face.h"
#include "Matrix.h"
#include <limits>



namespace TheWhiteAmbit {

	double Face::intersect(Ray ray)
	{
		// Find vectors for two edges sharing vert0
		D3DXVECTOR4 edge1 = v1.pos - v0.pos;
		D3DXVECTOR4 edge2 = v2.pos - v0.pos;

		// Begin calculating determinant - also used to calculate U parameter
		D3DXVECTOR3 pvec;
		//TODO: look if cast from (D3DXVECTOR4*) to (D3DXVECTOR3*) works fine
		D3DXVec3Cross( &pvec, (D3DXVECTOR3*)&ray.dir, (D3DXVECTOR3*)&edge2 );

		// If determinant is near zero, ray lies in plane of triangle
		FLOAT det = D3DXVec3Dot( (D3DXVECTOR3*)&edge1, &pvec );

		D3DXVECTOR4 tvec;
		if( det < 0 )
			return std::numeric_limits<double>::infinity();
		tvec = ray.orig - v0.pos;

		//Without culling
		//if( det > 0 )
		//    tvec = ray.orig - v0.pos;
		//else {
		//	tvec = v0.pos - ray.orig;
		//	det=-det;
		//}

		// Calculate U parameter and test bounds
		//TODO: look if cast from (D3DXVECTOR4*) to (D3DXVECTOR3*) works fine
		FLOAT u = D3DXVec3Dot( (D3DXVECTOR3*)&tvec, &pvec );
		if( u < 0.0f || u > det )
			return std::numeric_limits<double>::infinity();

		// Prepare to test V parameter
		D3DXVECTOR3 qvec;
		D3DXVec3Cross( &qvec, (D3DXVECTOR3*)&tvec, (D3DXVECTOR3*)&edge1 );

		// Calculate V parameter and test bounds
		//TODO: look if cast from (D3DXVECTOR4*) to (D3DXVECTOR3*) works fine
		FLOAT v = D3DXVec3Dot( (D3DXVECTOR3*)&ray.dir, &qvec );
		if( v < 0.0f || u + v > det )
			return std::numeric_limits<double>::infinity();

		// Calculate t, scale parameters, ray intersects triangle
		//TODO: look if cast from (D3DXVECTOR4*) to (D3DXVECTOR3*) works fine
		FLOAT t = D3DXVec3Dot( (D3DXVECTOR3*)&edge2, &qvec );
		FLOAT fInvDet = 1.0f / det;

		return t * fInvDet;
	}
}