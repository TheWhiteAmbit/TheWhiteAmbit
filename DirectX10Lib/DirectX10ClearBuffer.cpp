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




#include <math.h>
#include "DirectX10ClearBuffer.h"
#include "../SceneGraphLib/RenderVisitor.h"
namespace TheWhiteAmbit {
	DirectX10ClearBuffer::DirectX10ClearBuffer(DirectX10Renderer* a_pRenderer)
	{
		Node::Node();
		m_pRenderer = a_pRenderer;
	}

	DirectX10ClearBuffer::~DirectX10ClearBuffer(void)
	{
	}

	void DirectX10ClearBuffer::render(void)
	{
		float ClearColor[4] = { 0.5f, 0.5f, 0.5f, 1.0f }; //red,green,blue,alpha

		ClearColor[0]=(float)pow((double)(ClearColor[0]+0.055)/1.055, 2.4); //TODO: Make some SRGB translation math
		ClearColor[1]=(float)pow((double)(ClearColor[1]+0.055)/1.055, 2.4);
		ClearColor[2]=(float)pow((double)(ClearColor[2]+0.055)/1.055, 2.4);
		ClearColor[3]=(float)pow((double)(ClearColor[3]+0.055)/1.055, 2.4);

		m_pRenderer->getDevice()->ClearRenderTargetView( m_pRenderer->getRenderTargetView(), ClearColor );
		m_pRenderer->getDevice()->ClearDepthStencilView( m_pRenderer->getDepthStencilView(), D3D10_CLEAR_DEPTH , 1.0, 0);
	}

	void DirectX10ClearBuffer::accept(RenderVisitor* a_pRenderVisitor)
	{
		a_pRenderVisitor->visit(this);
		Node::accept(a_pRenderVisitor);
	}
}