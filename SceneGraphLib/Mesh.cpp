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



#include "Mesh.h"
#include "Face.h"
#include "Matrix.h"
#include <limits>

namespace TheWhiteAmbit {

	Mesh::Mesh(DWORD* a_pIndices, Vertex* a_pVertices, UINT a_iIndices, UINT a_iVertices)
	{
		m_pBoundingSphere=NULL;

		m_iIndices=a_iIndices;
		m_iVertices=a_iVertices;

		m_pIndices=new DWORD[a_iIndices];
		m_pVertices=new Vertex[a_iVertices];

		memcpy(m_pIndices, a_pIndices, a_iIndices * sizeof(DWORD));
		memcpy(m_pVertices, a_pVertices, a_iVertices * sizeof(Vertex));

		m_pUnitTriangles=NULL;
		m_iNumUnitTriangles=0;
	}

	Mesh::~Mesh(void)
	{
		delete[] m_pIndices;
		delete[] m_pVertices;
		if(m_pBoundingSphere)
			delete m_pBoundingSphere;
		if(m_pUnitTriangles)
			delete m_pUnitTriangles;
	}


	DWORD* Mesh::getIndices(void)
	{
		return m_pIndices;
	}
	UINT Mesh::getNumIndices(void)
	{
		return m_iIndices;
	}
	Vertex* Mesh::getVertices(void)
	{
		return m_pVertices;
	}
	UINT Mesh::getNumVertices(void)
	{
		return m_iVertices;
	}

	D3DXMATRIX*	Mesh::getUnitTriangles(void)
	{
		if(!this->m_pUnitTriangles)
			generateUnitTriangles();
		return this->m_pUnitTriangles;
	}

	unsigned int Mesh::getNumUnitTriangles(void)
	{
		if(!this->m_pUnitTriangles)
			generateUnitTriangles();
		return this->m_iNumUnitTriangles;
	}


	Sphere Mesh::getBoundingSphere(void)
	{
		if(!m_pBoundingSphere)
			generateBoundingSphere();
		return *m_pBoundingSphere;
	}


	double Mesh::intersectTriangles(Ray a_Ray)
	{
		double retDist=std::numeric_limits<double>::infinity();
		Face triangle;
		unsigned int iFaces=this->m_iIndices/3;

		for(unsigned int i=0;i<m_iIndices;i+=3)
		{
			triangle.v0=m_pVertices[m_pIndices[i]];
			triangle.v1=m_pVertices[m_pIndices[i+1]];
			triangle.v2=m_pVertices[m_pIndices[i+2]];
			double triDist=triangle.intersect(a_Ray);
			if(triDist>=0.0)
				retDist=min(retDist,triDist);
		}
		return retDist;
	}

	double Mesh::intersectBound(Ray a_Ray)
	{
		return this->getBoundingSphere().intersect(a_Ray);
	}


	void Mesh::generateUnitTriangles(void)
	{
		Face triangle;

		if(this->m_pUnitTriangles)
			delete[] this->m_pUnitTriangles;
		m_iNumUnitTriangles=0;
		this->m_pUnitTriangles=new D3DXMATRIX[m_iIndices/3];

		for(unsigned int i=0;i<m_iIndices;i+=3)
		{
			triangle.v0=m_pVertices[m_pIndices[i]];
			triangle.v1=m_pVertices[m_pIndices[i+1]];
			triangle.v2=m_pVertices[m_pIndices[i+2]];
			D3DXMATRIX unitMat=
				Matrix::UnitTriangle(triangle.v0.pos, triangle.v1.pos, triangle.v2.pos);		
			if(D3DXMatrixInverse(&unitMat, NULL, &unitMat))
			{
				////TODO: transpose speed test for CUDA - transposed matrices seems minimal faster
				//D3DXMatrixTranspose(&unitMat, &unitMat);
				
				//Note: Doubled values
				//unitMat._13=unitMat._14;
				//unitMat._23=unitMat._24;
				//unitMat._33=unitMat._34;
				//unitMat._43=-1.0+unitMat._44;

				this->m_pUnitTriangles[m_iNumUnitTriangles]=unitMat;
				this->m_iNumUnitTriangles++;
			}
		}
	}

	void Mesh::generateBoundingSphere(void)
	{
		D3DXVECTOR4 center=D3DXVECTOR4(0,0,0,1);
		FLOAT radius=0.0;
		//TODO: look if cast from D3DXVECTOR4 to D3DXVECTOR3 is working fine
		D3DXComputeBoundingSphere(
			(D3DXVECTOR3*)&this->m_pVertices->pos,
			this->m_iVertices,
			sizeof(Vertex),
			(D3DXVECTOR3*)&center,
			&radius);
		m_pBoundingSphere=new Sphere(center, radius);
	}
}