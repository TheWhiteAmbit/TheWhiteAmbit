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




#include "HexahedronBisectionH.h"
#include "HexahedronBisectionV.h"
#include "HexahedronBisectionP.h"

namespace TheWhiteAmbit {
	HexahedronBisectionH::HexahedronBisectionH(Hexahedron a_Hexahedron, unsigned int a_iDepth)
		:AHexahedronBisection(a_Hexahedron, a_iDepth)
	{
		m_vSplitPoint0=(m_Hexahedron.v0+m_Hexahedron.v2)*.5;
		m_vSplitPoint1=(m_Hexahedron.v1+m_Hexahedron.v3)*.5;
		m_vSplitPoint2=(m_Hexahedron.v4+m_Hexahedron.v6)*.5;
		m_vSplitPoint3=(m_Hexahedron.v5+m_Hexahedron.v7)*.5;
		D3DXPlaneFromPoints(&m_SplitPlane, (D3DXVECTOR3*)&m_vSplitPoint0, (D3DXVECTOR3*)&m_vSplitPoint1, (D3DXVECTOR3*)&m_vSplitPoint2);
	}

	HexahedronBisectionH::~HexahedronBisectionH(void)
	{
		//TODO: free bisection mem
	}

	IBisectionStrategy*	HexahedronBisectionH::getLeftStrategy(void)
	{
		Hexahedron t( m_vSplitPoint0, m_vSplitPoint1, m_Hexahedron.v2, m_Hexahedron.v3, m_vSplitPoint2, m_vSplitPoint3, m_Hexahedron.v6, m_Hexahedron.v7);
		return new HexahedronBisectionP(t, m_iDepth+1);
	}

	IBisectionStrategy*	HexahedronBisectionH::getRightStrategy(void)
	{
		Hexahedron t(m_Hexahedron.v0, m_Hexahedron.v1, m_vSplitPoint0, m_vSplitPoint1, m_Hexahedron.v4, m_Hexahedron.v5, m_vSplitPoint2, m_vSplitPoint3);
		return new HexahedronBisectionP(t, m_iDepth+1);
	}
}