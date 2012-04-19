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
#include "DirectX10Renderer.h"
#include "DirectX10Texture.h"
namespace TheWhiteAmbit {
	class DirectX10TargetRender : public IPresentable
	{
		Node* m_pRootNode;
		unsigned long m_iTicks;
		DirectX10Renderer* m_pRenderer;

		DirectX10Texture* m_pDepthStencilTexture;
		DirectX10Texture* m_pRenderTargetTexture;
		ID3D10RenderTargetView* m_pRenderTargetView;
		ID3D10DepthStencilView* m_pDepthStencilView;

		unsigned int m_iWidth;
		unsigned int m_iHeight;
	public:
		void present(IEffect* a_pEffect);
		void setRootNode(Node* a_pNode);
		void activateRenderTarget(void);
		DirectX10Texture* getTexture(void);

		DirectX10TargetRender(DirectX10Renderer* a_pRenderer);
		~DirectX10TargetRender(void);
	};
}