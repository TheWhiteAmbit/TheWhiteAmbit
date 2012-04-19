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

#include "../SceneGraphLib/Node.h"
#include "../SceneGraphLib/IPresentable.h"
#include "DirectX9Renderer.h"
#include "DirectX9Texture.h"

namespace TheWhiteAmbit {
	class DirectX9TargetRender : public IPresentable
	{
		DirectX9Texture*	m_pRenderTexture;
		IDirect3DSurface9*	m_pDepthSurface;

		Node* m_pRootNode;
		unsigned long m_iTicks;
		DirectX9Renderer* m_pRenderer;

		unsigned int m_iWidth;
		unsigned int m_iHeight;
	public:
		void present(IEffect* a_pEffect);
		void setRootNode(Node* a_pNode);
		DirectX9TargetRender(DirectX9Renderer* a_pRenderer);
		DirectX9TargetRender(DirectX9Renderer* a_pRenderer, D3DFORMAT a_D3dFormat);
		DirectX9Texture*	getTexture(void);
		virtual ~DirectX9TargetRender(void);
	};
}
