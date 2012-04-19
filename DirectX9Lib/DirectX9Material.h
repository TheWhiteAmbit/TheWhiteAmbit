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
#include "DirectX9Effect.h"
#include "DirectX9Texture.h"
#include "../SceneGraphLib/Node.h"
#include "../SceneGraphLib/IRenderable.h"
#include "../SceneGraphLib/Vertex.h"
namespace TheWhiteAmbit {
	class DirectX9Material :
		public Node, IRenderable
	{
		DirectX9Texture*				m_pTexture;
		DirectX9Renderer*				m_pRenderer;
	public:
		DirectX9Material(DirectX9Renderer* a_pRenderer);
		virtual ~DirectX9Material(void);
		void setTextureSource(unsigned int a_iTextureNumber, DirectX9Texture* a_pTexture);
		DirectX9Texture* getTextureSource(unsigned int a_iTextureNumber);

		//IRenderable
		virtual void render(IEffect* a_pEffect);

		//Visitors
		virtual void accept(RenderVisitor* a_pRenderVisitor);
	};
}