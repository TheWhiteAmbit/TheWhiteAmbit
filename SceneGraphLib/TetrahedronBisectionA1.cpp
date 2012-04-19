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



#include "TetrahedronBisectionA1.h"
#include "TetrahedronBisectionB0.h"
#include "TetrahedronBisectionB1.h"

namespace TheWhiteAmbit {
	TetrahedronBisectionA1::TetrahedronBisectionA1(Tetrahedron a_Tetrahedron, unsigned int a_iDepth)
		:ATetrahedronBisection(a_Tetrahedron, a_iDepth)
	{
	}

	TetrahedronBisectionA1::~TetrahedronBisectionA1(void)
	{
		//TODO: free bisection mem
	}

	IBisectionStrategy*	TetrahedronBisectionA1::getLeftStrategy(void)
	{
		Tetrahedron t(m_Tetrahedron.v0, m_Tetrahedron.v2, m_vSplitPoint, m_Tetrahedron.v1);
		return new TetrahedronBisectionB0(t, m_iDepth+1);
	}

	IBisectionStrategy*	TetrahedronBisectionA1::getRightStrategy(void)
	{
		Tetrahedron t(m_Tetrahedron.v3, m_Tetrahedron.v1, m_vSplitPoint, m_Tetrahedron.v2);
		return new TetrahedronBisectionB0(t, m_iDepth+1);
	}
}