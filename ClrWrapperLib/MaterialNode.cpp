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

#include "MaterialNode.h"

using namespace System;

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		void MaterialNode::SetTexture(unsigned int a_iTextureNumber, TextureAsset^ a_pAsset)
		{
			DirectX9Material* textueLayerNode=(DirectX9Material*)this->m_pNode;
			textueLayerNode->setTextureSource(a_iTextureNumber, a_pAsset->GetUnmanagedAsset());
			switch(a_iTextureNumber) {
			case 0:
				this->m_pTexture=a_pAsset;
				break;
			default:
				break;
			}
		}

		MaterialNode::MaterialNode(DirectX9Renderer* a_pDevice)
		{
			this->m_pNode=new DirectX9Material(a_pDevice);
			this->m_pTexture=nullptr;
		}
	}
}