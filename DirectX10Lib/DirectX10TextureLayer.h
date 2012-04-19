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

#include "DirectX10Renderer.h"
#include "DirectX10Effect.h"
#include "DirectX10Texture.h"
#include "../SceneGraphLib/Node.h"
#include "../SceneGraphLib/IRenderable.h"
#include "../SceneGraphLib/Vertex.h"
namespace TheWhiteAmbit {
	class DirectX10TextureLayer :
		public Node, IRenderable
	{
		DirectX10Effect*				m_pEffect;
		DirectX10Texture*				m_pTexture;
		DirectX10Renderer*				m_pRenderer;

		ID3D10InputLayout*      m_pVertexLayout;
		ID3D10Buffer*           m_pVertexBuffer;
		ID3D10EffectTechnique*  m_pTechnique;
		ID3D10ShaderResourceView* m_pTextureResource;
		ID3D10EffectShaderResourceVariable* m_ptxDiffuseVariable;

		void fitVertexBufferTexcoords(void);
	public:
		DirectX10TextureLayer(DirectX10Renderer* a_pRenderer);
		virtual ~DirectX10TextureLayer(void);

		void setEffect(DirectX10Effect* a_pEffect);
		void setTexture(DirectX10Texture* a_pTexture);

		//IRenderable
		virtual void render(void);

		//Visitors
		virtual void accept(RenderVisitor* a_pRenderVisitor);
	};
}