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



#pragma once
#include <iostream>
#include "../DirectXUTLib/DXUT.h"
#include "Ray.h"

namespace TheWhiteAmbit {

	class Tetrahedron
	{
	public:
		Tetrahedron(D3DXVECTOR4 a_v0, D3DXVECTOR4 a_v1, D3DXVECTOR4 a_v2, D3DXVECTOR4 a_v3);
		Tetrahedron(const Tetrahedron& t);

		const D3DXVECTOR4 v0;
		const D3DXVECTOR4 v1;
		const D3DXVECTOR4 v2;
		const D3DXVECTOR4 v3;

		const D3DXMATRIX planeMatrix;

		double intersect(Ray &ray);
	};

}
