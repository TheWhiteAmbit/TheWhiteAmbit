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




#include "DirectX9Material.h"
//#include "../SceneGraphLib/RenderVisitor.h"

#define D3DFVF_CUSTOMVERTEX 0
namespace TheWhiteAmbit {
	DirectX9Material::DirectX9Material(DirectX9Renderer* a_pRenderer)
	{
		m_pRenderer=a_pRenderer;
		m_pTexture=NULL;
	}

	DirectX9Material::~DirectX9Material(void)
	{
	}

	//TODO: insert Effect code
	void DirectX9Material::render(IEffect* a_pEffect)
	{
		DirectX9Effect* pEffect = (DirectX9Effect*) a_pEffect->getDirectX9Effect();
		if(pEffect && pEffect->getEffect() && m_pTexture)
		{
			D3DXHANDLE ptxDiffuseVariable;
			ptxDiffuseVariable= pEffect->getEffect()->GetParameterByName( NULL, "g_MeshTexture" );
			pEffect->getEffect()->SetTexture( ptxDiffuseVariable, this->m_pTexture->getTexture() );		
		}
	}

	void DirectX9Material::setTextureSource(unsigned int a_iTextureNumber, DirectX9Texture* a_pTexture)
	{
		switch(a_iTextureNumber) {
		case 0:
			this->m_pTexture=a_pTexture;
			break;
		default:
			break;
		}
	}

	DirectX9Texture* DirectX9Material::getTextureSource(unsigned int a_iTextureNumber)
	{
		switch(a_iTextureNumber) {
		case 0:
			return this->m_pTexture;
			break;
		default:
			return NULL;
		}
	}

	void DirectX9Material::accept(RenderVisitor* a_pRenderVisitor)
	{
		a_pRenderVisitor->visit(this);
		Node::accept(a_pRenderVisitor);
	}
}