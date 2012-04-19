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




#include "TetrahedronBisectionC1.h"
#include "TetrahedronBisectionA1.h"
#include "TetrahedronBisectionA0.h"
namespace TheWhiteAmbit {
	TetrahedronBisectionC1::TetrahedronBisectionC1(Tetrahedron a_Tetrahedron, unsigned int a_iDepth)
		:ATetrahedronBisection(a_Tetrahedron, a_iDepth)
	{
	}

	TetrahedronBisectionC1::~TetrahedronBisectionC1(void)
	{
		//TODO: free bisection mem
	}

	IBisectionStrategy*	TetrahedronBisectionC1::getLeftStrategy(void)
	{
		Tetrahedron t(m_Tetrahedron.v0, m_Tetrahedron.v2, m_vSplitPoint, m_Tetrahedron.v1);
		return new TetrahedronBisectionA1(t, m_iDepth+1);
	}

	IBisectionStrategy*	TetrahedronBisectionC1::getRightStrategy(void)
	{
		Tetrahedron t(m_Tetrahedron.v3, m_vSplitPoint, m_Tetrahedron.v2, m_Tetrahedron.v1);
		return new TetrahedronBisectionA0(t, m_iDepth+1);
	}
}