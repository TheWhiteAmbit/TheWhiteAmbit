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

#include "TextureLayerNode.h"

using namespace System;

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		void TextureLayerNode::SetTexture(unsigned int a_iTextureNumber, TextureAsset^ a_pAsset)
		{
			DirectX9TextureLayer* textueLayerNode=(DirectX9TextureLayer*)this->m_pNode;
			textueLayerNode->setTextureSource(a_iTextureNumber, a_pAsset->GetUnmanagedAsset());
			switch(a_iTextureNumber) {
			case 0:
				this->m_pTexture=a_pAsset;
				break;
			default:
				break;
			}
		}

		void TextureLayerNode::SetYOrthogonalPosition(double minX, double minY, double maxX, double maxY)
		{
			DirectX9TextureLayer* textueLayerNode=(DirectX9TextureLayer*)this->m_pNode;
			textueLayerNode->setYOrthogonalPosition(minX, minY, maxX, maxY);
		}

		void TextureLayerNode::SetZOrthogonalPosition(double minX, double minY, double maxX, double maxY)
		{
			DirectX9TextureLayer* textueLayerNode=(DirectX9TextureLayer*)this->m_pNode;
			textueLayerNode->setZOrthogonalPosition(minX, minY, maxX, maxY);
		}

		TextureLayerNode::TextureLayerNode(DirectX9Renderer* a_pDevice)
		{
			this->m_pNode=new DirectX9TextureLayer(a_pDevice);
			this->m_pTexture=nullptr;
		}
	}
}