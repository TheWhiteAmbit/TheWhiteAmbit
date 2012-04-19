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

namespace TheWhiteAmbit {

	class Matrix
	{
	public:
		static FLOAT Minor(const D3DXMATRIX& m, 
			const char r0, const char r1, const char r2, 
			const char c0, const char c1, const char c2);

		static D3DXMATRIX Adjoint(const D3DXMATRIX& m);
		static D3DXMATRIX Inverse(const D3DXMATRIX& m);

		static D3DXMATRIX UnitTriangle(const D3DXVECTOR3& v0,const D3DXVECTOR3& v1,const D3DXVECTOR3& v2);
		static D3DXMATRIX UnitTriangle(const D3DXVECTOR4& v0,const D3DXVECTOR4& v1,const D3DXVECTOR4& v2);

		static D3DXMATRIX UnitTetrahedron(const D3DXVECTOR3& v0,const D3DXVECTOR3& v1,const D3DXVECTOR3& v2, const D3DXVECTOR3& v3);
		static D3DXMATRIX UnitTetrahedron(const D3DXVECTOR4& v0,const D3DXVECTOR4& v1,const D3DXVECTOR4& v2, const D3DXVECTOR4& v3);

		static D3DXMATRIX RowsFromVec4(const D3DXVECTOR4& v0,const D3DXVECTOR4& v1,const D3DXVECTOR4& v2,const D3DXVECTOR4& v3);
		static D3DXMATRIX ColumnsFromVec4(const D3DXVECTOR4& v0,const D3DXVECTOR4& v1,const D3DXVECTOR4& v2,const D3DXVECTOR4& v3);

		static D3DXMATRIX RowsFromVec3(const D3DXVECTOR3& v0,const D3DXVECTOR3& v1,const D3DXVECTOR3& v2,const D3DXVECTOR3& v3);
		static D3DXMATRIX RowsFromVec3Normal(const D3DXVECTOR3& v0,const D3DXVECTOR3& v1,const D3DXVECTOR3& v2,const D3DXVECTOR3& v3);
		static D3DXMATRIX RowsFromVec3Coord(const D3DXVECTOR3& v0,const D3DXVECTOR3& v1,const D3DXVECTOR3& v2,const D3DXVECTOR3& v3);
		static D3DXMATRIX RowsFromVec3Unit(const D3DXVECTOR3& v0,const D3DXVECTOR3& v1,const D3DXVECTOR3& v2,const D3DXVECTOR3& v3);

		static D3DXMATRIX ColumnsFromVec3(const D3DXVECTOR3& v0,const D3DXVECTOR3& v1,const D3DXVECTOR3& v2,const D3DXVECTOR3& v3);
		static D3DXMATRIX ColumnsFromVec3Normal(const D3DXVECTOR3& v0,const D3DXVECTOR3& v1,const D3DXVECTOR3& v2,const D3DXVECTOR3& v3);
		static D3DXMATRIX ColumnsFromVec3Coord(const D3DXVECTOR3& v0,const D3DXVECTOR3& v1,const D3DXVECTOR3& v2,const D3DXVECTOR3& v3);
	};

}
