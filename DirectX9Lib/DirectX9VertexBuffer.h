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

#include "DirectX9Renderer.h"
#include "../SceneGraphLib/Mesh.h"
namespace TheWhiteAmbit {
	class DirectX9VertexBuffer
	{
		IDirect3DVertexBuffer9* m_pVertexBuffer;
		DirectX9Renderer*		m_pRenderer;
		unsigned int			m_iVertexCount;
	public:
		DirectX9VertexBuffer(DirectX9Renderer* a_pRenderer);
		~DirectX9VertexBuffer(void);

		void setMesh(Mesh*	a_pMesh);
		void setVertices(Vertex* a_pVertices, unsigned int a_iNumVertices);

		IDirect3DVertexBuffer9* getVertexBuffer();
		unsigned int getNumVertices();

		void Render(ID3DXEffect* a_pEffect);
	};
}
