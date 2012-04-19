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




#include "Matrix.h"


namespace TheWhiteAmbit {

	FLOAT Matrix::Minor(const D3DXMATRIX& m, 
		const char row0, const char row1, const char row2, 
		const char col0, const char col1, const char col2)
	{
		return  m(row0,col0) * (m(row1,col1) * m(row2,col2) - m(row2,col1) * m(row1,col2)) -
			m(row0,col1) * (m(row1,col0) * m(row2,col2) - m(row2,col0) * m(row1,col2)) +
			m(row0,col2) * (m(row1,col0) * m(row2,col1) - m(row2,col0) * m(row1,col1));
	}


	D3DXMATRIX Matrix::Adjoint(const D3DXMATRIX& m)
	{
		return D3DXMATRIX(
			Minor(m, 1, 2, 3, 1, 2, 3),
			-Minor(m, 0, 2, 3, 1, 2, 3),
			Minor(m, 0, 1, 3, 1, 2, 3),
			-Minor(m, 0, 1, 2, 1, 2, 3),

			-Minor(m, 1, 2, 3, 0, 2, 3),
			Minor(m, 0, 2, 3, 0, 2, 3),
			-Minor(m, 0, 1, 3, 0, 2, 3),
			Minor(m, 0, 1, 2, 0, 2, 3),

			Minor(m, 1, 2, 3, 0, 1, 3),
			-Minor(m, 0, 2, 3, 0, 1, 3),
			Minor(m, 0, 1, 3, 0, 1, 3),
			-Minor(m, 0, 1, 2, 0, 1, 3),

			-Minor(m, 1, 2, 3, 0, 1, 2),
			Minor(m, 0, 2, 3, 0, 1, 2),
			-Minor(m, 0, 1, 3, 0, 1, 2),
			Minor(m, 0, 1, 2, 0, 1, 2));
	}

	D3DXMATRIX Matrix::Inverse(const D3DXMATRIX& m)
	{
		D3DXMATRIX inv;
		D3DXMatrixInverse(&inv, NULL, &m);
		return inv;
	}

	D3DXMATRIX Matrix::UnitTriangle(const D3DXVECTOR3& v0, const D3DXVECTOR3& v1, const D3DXVECTOR3& v2)
	{
		D3DXVECTOR3 norm0;
		//D3DXVec3Cross(&norm0, &(v0-v2), &(v0-v1));
		D3DXVec3Cross(&norm0, &(v0-v1), &(v0-v2));
		D3DXVec3Normalize(&norm0, &norm0);
		return RowsFromVec3Unit(v0-v2, v1-v2, norm0-v2, v2);
	}

	D3DXMATRIX Matrix::UnitTriangle(const D3DXVECTOR4& v0, const D3DXVECTOR4& v1, const D3DXVECTOR4& v2)
	{
		return UnitTriangle((D3DXVECTOR3)v0, (D3DXVECTOR3)v1, (D3DXVECTOR3)v2);
	}

	D3DXMATRIX Matrix::UnitTetrahedron(const D3DXVECTOR3& v0, const D3DXVECTOR3& v1, const D3DXVECTOR3& v2, const D3DXVECTOR3& v3)
	{
		return RowsFromVec3(v0-v2, v1-v2, v3-v2, v2);
	}

	D3DXMATRIX Matrix::UnitTetrahedron(const D3DXVECTOR4& v0, const D3DXVECTOR4& v1, const D3DXVECTOR4& v2, const D3DXVECTOR4& v3)
	{
		return UnitTetrahedron((D3DXVECTOR3)v0, (D3DXVECTOR3)v1, (D3DXVECTOR3)v2, (D3DXVECTOR3)v3);
	}

	D3DXMATRIX Matrix::RowsFromVec4(
		const D3DXVECTOR4& v0,
		const D3DXVECTOR4& v1,
		const D3DXVECTOR4& v2,
		const D3DXVECTOR4& v3)
	{
		return D3DXMATRIX(
			v0.x, v0.y, v0.z, v0.w,
			v1.x, v1.y, v1.z, v1.w,
			v2.x, v2.y, v2.z, v2.w,
			v3.x, v3.y, v3.z, v3.w);
	}

	D3DXMATRIX Matrix::ColumnsFromVec4(
		const D3DXVECTOR4& v0,
		const D3DXVECTOR4& v1,
		const D3DXVECTOR4& v2,
		const D3DXVECTOR4& v3)
	{
		return D3DXMATRIX(
			v0.x, v1.x, v2.x, v3.x,
			v0.y, v1.y, v2.y, v3.y,
			v0.z, v1.z, v2.z, v3.z,
			v0.w, v1.w, v2.w, v3.w);
	}

	D3DXMATRIX Matrix::RowsFromVec3(
		const D3DXVECTOR3& v0,
		const D3DXVECTOR3& v1,
		const D3DXVECTOR3& v2,
		const D3DXVECTOR3& v3)
	{
		return D3DXMATRIX(
			v0.x, v0.y, v0.z, 0.0f,
			v1.x, v1.y, v1.z, 0.0f,
			v2.x, v2.y, v2.z, 0.0f,
			v3.x, v3.y, v3.z, 1.0f);
	}

	D3DXMATRIX Matrix::RowsFromVec3Normal(
		const D3DXVECTOR3& v0,
		const D3DXVECTOR3& v1,
		const D3DXVECTOR3& v2,
		const D3DXVECTOR3& v3)
	{
		return D3DXMATRIX(
			v0.x, v0.y, v0.z, 0.0f,
			v1.x, v1.y, v1.z, 0.0f,
			v2.x, v2.y, v2.z, 0.0f,
			v3.x, v3.y, v3.z, 0.0f);
	}

	D3DXMATRIX Matrix::RowsFromVec3Coord(
		const D3DXVECTOR3& v0,
		const D3DXVECTOR3& v1,
		const D3DXVECTOR3& v2,
		const D3DXVECTOR3& v3)
	{
		return D3DXMATRIX(
			v0.x, v0.y, v0.z, 1.0f,
			v1.x, v1.y, v1.z, 1.0f,
			v2.x, v2.y, v2.z, 1.0f,
			v3.x, v3.y, v3.z, 1.0f);
	}

	D3DXMATRIX Matrix::RowsFromVec3Unit(
		const D3DXVECTOR3& v0,
		const D3DXVECTOR3& v1,
		const D3DXVECTOR3& v2,
		const D3DXVECTOR3& v3)
	{
		return D3DXMATRIX(
			v0.x, v0.y, v0.z, 0.0f,
			v1.x, v1.y, v1.z, 0.0f,
			v2.x, v2.y, v2.z, -1.0f,
			v3.x, v3.y, v3.z, 1.0f);
	}

	D3DXMATRIX Matrix::ColumnsFromVec3(
		const D3DXVECTOR3& v0,
		const D3DXVECTOR3& v1,
		const D3DXVECTOR3& v2,
		const D3DXVECTOR3& v3)
	{
		return D3DXMATRIX(
			v0.x, v1.x, v2.x, v3.x,
			v0.y, v1.y, v2.y, v3.y,
			v0.z, v1.z, v2.z, v3.z,
			0.0f, 0.0f, 0.0f, 1.0f);
	}

	D3DXMATRIX Matrix::ColumnsFromVec3Normal(
		const D3DXVECTOR3& v0,
		const D3DXVECTOR3& v1,
		const D3DXVECTOR3& v2,
		const D3DXVECTOR3& v3)
	{
		return D3DXMATRIX(
			v0.x, v1.x, v2.x, v3.x,
			v0.y, v1.y, v2.y, v3.y,
			v0.z, v1.z, v2.z, v3.z,
			0.0f, 0.0f, 0.0f, 0.0f);
	}

	D3DXMATRIX Matrix::ColumnsFromVec3Coord(
		const D3DXVECTOR3& v0,
		const D3DXVECTOR3& v1,
		const D3DXVECTOR3& v2,
		const D3DXVECTOR3& v3)
	{
		return D3DXMATRIX(
			v0.x, v1.x, v2.x, v3.x,
			v0.y, v1.y, v2.y, v3.y,
			v0.z, v1.z, v2.z, v3.z,
			1.0f, 1.0f, 1.0f, 1.0f);
	}
}
