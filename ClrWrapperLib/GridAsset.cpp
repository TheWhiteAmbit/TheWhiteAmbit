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



#include "stdafx.h"

#include "GridAsset.h"

using namespace System;

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		Grid<Color>* GridAsset::GetUnmanagedAsset(void)
		{
			return this->m_pGrid;
		}

		GridAsset::GridAsset(Grid<Color>* a_pGrid)
		{
			this->m_pGrid=a_pGrid;
		}

		GridAsset^ GridAsset::GetSubGrid(unsigned int minX, unsigned int minY, unsigned int maxX, unsigned int maxY)
		{
			unsigned int diffX = abs( (long)maxX - (long)minX );
			unsigned int diffY = abs( (long)maxY - (long)minY );
			Grid<Color>* pGrid=new Grid<Color>(diffX, diffY);
			for (unsigned int y=0; y<diffY; y++)
			{
				for (unsigned int x=0; x<diffX; x++)
				{
					pGrid->setPixel(x, y, m_pGrid->getPixel(x+minX, y+minY));
				}
			}
			return gcnew GridAsset(pGrid);
		}

		void GridAsset::Release(){
			delete this->m_pGrid;
		}
	}
}