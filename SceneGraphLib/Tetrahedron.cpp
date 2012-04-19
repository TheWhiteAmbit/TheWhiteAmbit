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



#include "Tetrahedron.h"
#include "Matrix.h"
#include <limits>

namespace TheWhiteAmbit {

	Tetrahedron::Tetrahedron(D3DXVECTOR4 a_v0, D3DXVECTOR4 a_v1, D3DXVECTOR4 a_v2, D3DXVECTOR4 a_v3)
		:	v0(a_v0),
		v1(a_v1),
		v2(a_v2),
		v3(a_v3),
		planeMatrix(Matrix::Adjoint(Matrix::RowsFromVec4(v0, v1, v2, v3)))
	{
		//Fill the Rows of the Matrix with Vertices, and Adjoint, since D3D uses transposed RowColumn order
	}

	Tetrahedron::Tetrahedron(const Tetrahedron& t)
		:v0(t.v0), v1(t.v1), v2(t.v2), v3(t.v3), planeMatrix(t.planeMatrix)
	{
	}

	double Tetrahedron::intersect(Ray &ray)
	{
		D3DXVECTOR4 vecOrig;
		D3DXVECTOR4 vecDir;
		D3DXVec4Transform(&vecOrig, &ray.orig, &this->planeMatrix);
		D3DXVec4Transform(&vecDir, &ray.dir, &this->planeMatrix);
		D3DXVECTOR4 vecHits;

		//distance valuese are inverted, this inverts the min/max below
		vecHits.x=vecOrig.x/vecDir.x; 
		vecHits.y=vecOrig.y/vecDir.y;
		vecHits.z=vecOrig.z/vecDir.z;
		vecHits.w=vecOrig.w/vecDir.w;

		//take the farthest distance to an entry point
		float entryPointNeg=std::numeric_limits<float>::infinity();
		//and the closest distance to an exit point (this defines the extent of the intersection).
		float exitPointNeg=-std::numeric_limits<float>::infinity();

		if(vecDir.x<0.0)	entryPointNeg=vecHits.x;
		else				exitPointNeg=vecHits.x;

		if(vecDir.y<0.0)	entryPointNeg=min(entryPointNeg, vecHits.y);
		else				exitPointNeg =max(exitPointNeg,  vecHits.y);

		if(vecDir.z<0.0)	entryPointNeg=min(entryPointNeg, vecHits.z);
		else				exitPointNeg =max(exitPointNeg,  vecHits.z);

		if(vecDir.w<0.0)	entryPointNeg=min(entryPointNeg, vecHits.w);
		else				exitPointNeg =max(exitPointNeg,  vecHits.w);

		//If the exit is beyond the entry point, there is an intersection
		double cutLength=entryPointNeg-exitPointNeg;
		return cutLength;
	}

	////TODO: new tetra test
	//double Tetrahedron::intersect(Ray &a_Ray)
	//{
	//	double result=std::numeric_limits<double>::infinity();
	//	D3DXVECTOR4 vecOrig;
	//	D3DXVECTOR4 vecDir;
	//
	//	D3DXVec4Transform(&vecOrig, &a_Ray.orig, &this->unitMatrix);
	//	D3DXVec4Transform(&vecDir, &a_Ray.dir, &this->unitMatrix);
	//
	//	{
	//		float tz=-vecOrig.z/vecDir.z;
	//		float uz=vecOrig.x+tz*vecDir.x;
	//		float vz=vecOrig.y+tz*vecDir.y;
	//		if(vz<0.0f || uz<0.0f || uz+vz>1.0f)
	//			return 1.0;
	//	}
	//	{
	//		float ty=-vecOrig.y/vecDir.y;
	//		float uy=vecOrig.x+ty*vecDir.x;
	//		float vy=vecOrig.z+ty*vecDir.z;
	//		if(vy<0.0f || uy<0.0f || uy+vy>1.0f)
	//			return 1.0;
	//	}
	//	//{
	//	//	float tx=-vecOrig.x/vecDir.x;
	//	//	float ux=vecOrig.z+tx*vecDir.z;
	//	//	float vx=vecOrig.y+tx*vecDir.y;
	//	//	if(vx<0.0f || ux<0.0f || ux+vx>1.0f)
	//	//		return 1.0;
	//	//}
	//	return -std::numeric_limits<float>::infinity();
	//}
}