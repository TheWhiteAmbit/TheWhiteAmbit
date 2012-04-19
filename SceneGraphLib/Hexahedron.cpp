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



#include "Hexahedron.h"
#include "Matrix.h"
#include <limits>

namespace TheWhiteAmbit {

	Hexahedron::Hexahedron(D3DXVECTOR4 a_v0, D3DXVECTOR4 a_v1, D3DXVECTOR4 a_v2, D3DXVECTOR4 a_v3, D3DXVECTOR4 a_v4, D3DXVECTOR4 a_v5, D3DXVECTOR4 a_v6, D3DXVECTOR4 a_v7)
		:v0(a_v0), v1(a_v1), v2(a_v2), v3(a_v3), v4(a_v4), v5(a_v5), v6(a_v6), v7(a_v7)
	{
	}

	Hexahedron::Hexahedron(const Hexahedron& t)
		:v0(t.v0), v1(t.v1), v2(t.v2), v3(t.v3), v4(t.v4), v5(t.v5), v6(t.v6), v7(t.v7)
	{
	}

	//float intersect(Ray ray, ConvexPoly poly){
	//	float t_in=0.0f;
	//	float t_out=std::numeric_limits<double>::infinity();
	//
	//	for(unsigned int i=poly.count; i>0; i--)	{
	//		float orig=D3DXPlaneDot(&poly.plane[i], &ray.orig);
	//		float dir=D3DXPlaneDot(&poly.plane[i], &ray.dir);
	//		float t_cut=-orig/dir;
	//		if(dir<0.0f)	t_in=max(t_in, t_cut);
	//		else			t_out=min(t_out, t_cut);
	//	}
	//	return t_out-t_in;
	//}

}