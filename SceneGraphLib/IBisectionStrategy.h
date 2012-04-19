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
	class IBisectionStrategy
	{
	public:
		virtual unsigned int getBisectionDepth(void)=0;
		virtual IBisectionStrategy*	getLeftStrategy(void)=0;
		virtual IBisectionStrategy*	getRightStrategy(void)=0;
		virtual D3DXPLANE  getSplitPlane(void)=0;
		virtual D3DXPLANE* getBoundingPlanes(unsigned int* pCount)=0;
	};
}