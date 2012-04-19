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
#include "BaseNode.h"
#include "../SceneGraphLib/Grid.h"
#include "../SceneGraphLib/Color.h"

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		public ref class GridAsset
		{
			Grid<Color>* m_pGrid;
		internal:		
			Grid<Color>* GetUnmanagedAsset(void);
		public:
			GridAsset(Grid<Color>* a_pGrid);
			GridAsset^ GetSubGrid(unsigned int minX, unsigned int minY, unsigned int maxX, unsigned int maxY);
			void Release();
		};
	}
}