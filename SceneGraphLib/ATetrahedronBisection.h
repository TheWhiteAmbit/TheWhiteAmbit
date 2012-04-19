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
#include "IBisectionStrategy.h"
#include "Tetrahedron.h"
namespace TheWhiteAmbit {
	class ATetrahedronBisection :
		public IBisectionStrategy
	{
	protected:
		Tetrahedron m_Tetrahedron;
		unsigned int m_iDepth;
		D3DXVECTOR4	m_vSplitPoint;
		D3DXPLANE	m_SplitPlane;
		D3DXPLANE*	m_pBoundingPlanes;
	public:
		ATetrahedronBisection(Tetrahedron a_Tetrahedron, unsigned int a_iDepth);
		~ATetrahedronBisection(void);

		virtual unsigned int getBisectionDepth(void);
		virtual D3DXPLANE  getSplitPlane(void);
		virtual D3DXPLANE* getBoundingPlanes(unsigned int* pCount);
	};
}
