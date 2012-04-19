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




#include "DirectX9ClearBuffer.h"
#include "../SceneGraphLib/RenderVisitor.h"
namespace TheWhiteAmbit {
	DirectX9ClearBuffer::DirectX9ClearBuffer(DirectX9Renderer* a_pRenderer)
	{
		Node::Node();
		m_pRenderer = a_pRenderer;
	}

	DirectX9ClearBuffer::~DirectX9ClearBuffer(void)
	{
	}

	//TODO: insert Effect code
	void DirectX9ClearBuffer::render(IEffect* a_pEffect)
	{
		//float ClearColor[4] = { 32.0/256.0, 32.0/256.0, 128.0/256.0, 1.0 }; //red,green,blue,alpha
		float ClearColor[4] = { .5, .5, .5, 0.0 }; //red,green,blue,alpha 
		//float ClearColor[4] = { .5, .5, .5, .5 }; //red,green,blue,alpha
		//float ClearColor[4] = { .0, .0, .0, .0 }; //red,green,blue,alpha
		//float ClearColor[4] = { .5, .5, .625, 1.0 }; //red,green,blue,alpha

		//float ClearColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f }; //red,green,blue,alpha
		//float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f }; //red,green,blue,alpha
		//float ClearColor[4] = { 0.8f, 0.8f, 0.8f, 1.0f }; //red,green,blue,alpha
		//float ClearColor[4] = { 0.0f, 0.125f, 0.3f, 1.0f }; //red,green,blue,alpha
		//m_pRenderer->getDevice()->Clear( 0, NULL, D3DCLEAR_ZBUFFER, D3DCOLOR_ARGB( 0xff, 0, 0, 255 ), 1.0f, 0 );
		m_pRenderer->getDevice()->Clear( 0, 
			NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER,
			D3DCOLOR_RGBA(
			(DWORD)(ClearColor[0]*255.0f),
			(DWORD)(ClearColor[1]*255.0f),
			(DWORD)(ClearColor[2]*255.0f),
			(DWORD)(ClearColor[3]*255.0f)),
			1.0f, 0 );	
		//m_pRenderer->getDevice()->Clear( 0, NULL, D3DCLEAR_TARGET, D3DCOLOR_XRGB( 0, 0, 255 ), 1.0f, 0 );
	}

	void DirectX9ClearBuffer::accept(RenderVisitor* a_pRenderVisitor)
	{
		a_pRenderVisitor->visit(this);
		Node::accept(a_pRenderVisitor);
	}
}