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

#include "Vertex.h"
#include "Ray.h"
#include "Sphere.h"

namespace TheWhiteAmbit {

	class Mesh
	{
	protected:
		DWORD* m_pIndices;
		Vertex* m_pVertices;
		UINT	m_iIndices;
		UINT	m_iVertices;
		Sphere*	m_pBoundingSphere;

		D3DXMATRIX*	m_pUnitTriangles;
		unsigned int m_iNumUnitTriangles;
		virtual void generateUnitTriangles(void);
		virtual void generateBoundingSphere(void);
	public:
		Mesh(DWORD* a_pIndices, Vertex* a_pVertices, UINT a_iIndices, UINT a_iVertices);	
		virtual ~Mesh(void);

		virtual DWORD* getIndices(void);
		virtual UINT getNumIndices(void);

		virtual Vertex* getVertices(void);
		virtual UINT getNumVertices(void);

		D3DXMATRIX*	getUnitTriangles(void);
		unsigned int getNumUnitTriangles(void);

		virtual Sphere getBoundingSphere(void);

		double intersectTriangles(Ray a_pRay);
		double intersectBound(Ray);

	};
}