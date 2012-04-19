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

	struct Vertex
	{
	public:
		D3DXVECTOR4 pos;
		D3DXVECTOR4 norm;
		D3DXVECTOR2 tex;
		static IDirect3DVertexDeclaration9* getVertexLayout(void);
		static D3DVERTEXELEMENT9* getVertexElements(void);
		static D3D10_INPUT_ELEMENT_DESC* getVertexElements(UINT* a_pNumElements);
		static DWORD getVertexFVF(void);
	};

}