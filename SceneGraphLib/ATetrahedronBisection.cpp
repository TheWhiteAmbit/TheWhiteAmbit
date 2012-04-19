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




#include "ATetrahedronBisection.h"
namespace TheWhiteAmbit {
	ATetrahedronBisection::ATetrahedronBisection(Tetrahedron a_Tetrahedron, unsigned int a_iDepth)
		:m_Tetrahedron(a_Tetrahedron)
	{
		m_iDepth=a_iDepth;
		m_vSplitPoint=(m_Tetrahedron.v0+m_Tetrahedron.v3)*.5;
		m_vSplitPoint.w=1.0;
		D3DXPlaneFromPoints(&m_SplitPlane, (D3DXVECTOR3*)&m_Tetrahedron.v1, (D3DXVECTOR3*)&m_Tetrahedron.v2, (D3DXVECTOR3*)&m_vSplitPoint);
		D3DXVECTOR4 norm=m_Tetrahedron.v0-m_vSplitPoint;
		D3DXVec3Normalize((D3DXVECTOR3*)&norm, (D3DXVECTOR3*)&norm);

		//TODO: uncomment to test normal direction correctness
		//float dir=D3DXPlaneDotNormal(&m_SplitPlane, (D3DXVECTOR3*)&norm);
		//if(dir<0.0f)
		//	D3DXPlaneFromPoints(&m_SplitPlane, (D3DXVECTOR3*)&m_Tetrahedron.v2, (D3DXVECTOR3*)&m_Tetrahedron.v1, (D3DXVECTOR3*)&m_vSplitPoint);

		m_pBoundingPlanes=NULL;
	}

	ATetrahedronBisection::~ATetrahedronBisection(void)
	{
		//TODO: free bisection mem
	}

	unsigned int ATetrahedronBisection::getBisectionDepth(void)
	{
		return this->m_iDepth;
	}

	D3DXPLANE	ATetrahedronBisection::getSplitPlane(void)
	{
		return this->m_SplitPlane;
	}

	D3DXPLANE*	ATetrahedronBisection::getBoundingPlanes(unsigned int* pCount)
	{
		if(!m_pBoundingPlanes)
		{
			D3DXMATRIX* pPlanesMatrix=new D3DXMATRIX;
			D3DXMatrixTranspose(pPlanesMatrix, &m_Tetrahedron.planeMatrix);
			m_pBoundingPlanes=(D3DXPLANE*)pPlanesMatrix;
		}
		*pCount=4; //number of bounding planes hardcoded
		return m_pBoundingPlanes;
	}
}