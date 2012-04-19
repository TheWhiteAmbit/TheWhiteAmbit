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


#include "Sphere.h"

namespace TheWhiteAmbit {

	Sphere::Sphere(D3DXVECTOR4 a_Center, FLOAT a_Radius)
		:center(a_Center), radius(a_Radius), radiusSq(a_Radius*a_Radius)
	{
	}

	Sphere::Sphere(const Sphere& s)
		:center(s.center), radius(s.radius), radiusSq(s.radius*s.radius)
	{
	}


	double Sphere::intersect(const Ray &ray)
	{
		D3DXVECTOR4 dst = ray.orig - center;
		FLOAT b = D3DXVec4Dot(&dst, &ray.dir);
		FLOAT c = D3DXVec4LengthSq(&dst) - radiusSq;
		return b*b - c;
		//FLOAT d=b*b - c;
		//return d > 0 ? -b - sqrt(d) : std::numeric_limits<float>::infinity();
	}

	Sphere& Sphere::operator=(Sphere const& other)
	{
		return *this;
	}
}