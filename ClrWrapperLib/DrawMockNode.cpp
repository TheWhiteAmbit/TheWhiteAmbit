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

#include "DrawMockNode.h"

using namespace System;

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		void DrawMockNode::SetPoints(
			double a0, double b0, double c0,
			double a1, double b1, double c1,
			double a2, double b2, double c2)
		{
			DirectX9DrawMock* drawMockNode=(DirectX9DrawMock*)this->m_pNode;
			drawMockNode->setPoints(
				D3DXVECTOR4((float)a0, (float)b0, (float)c0, 1.0f ), 
				D3DXVECTOR4((float)a1, (float)b1, (float)c1, 1.0f ), 
				D3DXVECTOR4((float)a2, (float)b2, (float)c2, 1.0f ));
		}

		void DrawMockNode::SetMesh(SdkMeshAsset^ a_pAsset)
		{
			DirectX9DrawMock* drawMockNode=(DirectX9DrawMock*)this->m_pNode;
			drawMockNode->setMesh(a_pAsset->GetUnmanagedAsset());
		}

		void DrawMockNode::SetMesh(ObjMeshAsset^ a_pAsset)
		{
			DirectX9DrawMock* drawMockNode=(DirectX9DrawMock*)this->m_pNode;
			drawMockNode->setMesh(a_pAsset->GetUnmanagedAsset());
		}

		DrawMockNode::DrawMockNode(DirectX9Renderer* a_pDevice)
		{
			this->m_pNode=new DirectX9DrawMock(a_pDevice);
		}
	}
}