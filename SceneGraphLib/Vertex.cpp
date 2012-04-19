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




#include "Vertex.h"

namespace TheWhiteAmbit {

	IDirect3DVertexDeclaration9* Vertex::getVertexLayout(void)
	{
		return NULL;
	}

	D3DVERTEXELEMENT9* Vertex::getVertexElements(void)
	{
		static D3DVERTEXELEMENT9 layout[] =
		{
			{0, 0, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
			{0, 16, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL, 0},
			{0, 32, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0},
			//{0, 40, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TANGENT, 0},
			D3DDECL_END() // this macro is needed as the last item!
		};
		return &layout[0];
	}

	D3D10_INPUT_ELEMENT_DESC* Vertex::getVertexElements(UINT* a_pNumElements)
	{
		static D3D10_INPUT_ELEMENT_DESC layout[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0 },
			{ "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 16, D3D10_INPUT_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 32, D3D10_INPUT_PER_VERTEX_DATA, 0 },
			//{ "TANGENT", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 40, D3D10_INPUT_PER_VERTEX_DATA, 0 },
		};
		*a_pNumElements = sizeof( layout ) / sizeof( layout[0] );
		return &layout[0];
	}

	//TODO: make D3DFVF_NORMAL a D3DXVECTOR4 but this is not used anyway
	DWORD Vertex::getVertexFVF(void)
	{
		return D3DFVF_XYZW | D3DFVF_NORMAL | D3DFVF_TEX0;
	}

}