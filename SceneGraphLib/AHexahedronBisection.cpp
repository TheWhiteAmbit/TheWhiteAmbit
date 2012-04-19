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




#include "AHexahedronBisection.h"
namespace TheWhiteAmbit {
	AHexahedronBisection::AHexahedronBisection(Hexahedron a_Hexahedron, unsigned int a_iDepth)
		:m_Hexahedron(a_Hexahedron)
	{
		m_iDepth=a_iDepth;
		m_pBoundingPlanes=NULL;
	}

	AHexahedronBisection::~AHexahedronBisection(void)
	{
		//TODO: free bisection mem
	}

	unsigned int AHexahedronBisection::getBisectionDepth(void)
	{
		return this->m_iDepth;
	}

	D3DXPLANE	AHexahedronBisection::getSplitPlane(void)
	{
		return this->m_SplitPlane;
	}

	D3DXPLANE*	AHexahedronBisection::getBoundingPlanes(unsigned int* pCount)
	{
		if(!m_pBoundingPlanes)
		{
			D3DXPLANE* pPlanesMatrix=new D3DXPLANE[6];
			m_pBoundingPlanes=(D3DXPLANE*)pPlanesMatrix;
			//TODO: generate planes here
		}
		*pCount=6; //number of bounding planes hardcoded
		return m_pBoundingPlanes;
	}
}