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



#include "DirectX10ViewRender.h"
#include "../SceneGraphLib/RenderVisitor.h"
#include "../SceneGraphLib/TransformVisitor.h"
#include "../SceneGraphLib/PickVisitor.h"
namespace TheWhiteAmbit {
	DirectX10ViewRender::DirectX10ViewRender(DirectX10Renderer* a_pRenderer)
	{
		m_pRenderer=NULL;
		m_pRootNode=NULL;
		m_iTicks=0;
		m_pRenderer=a_pRenderer;
	}

	DirectX10ViewRender::~DirectX10ViewRender(void)
	{
	}

	//--------------------------------------------------------------------------------------
	// Render the frame
	//--------------------------------------------------------------------------------------
	void DirectX10ViewRender::present(IEffect* a_pEffect)
	{
		m_iTicks++;
		RenderVisitor renderNodeVisitor(a_pEffect);
		if(m_pRootNode)
			m_pRootNode->accept(&renderNodeVisitor);

		if(m_pRenderer)
			m_pRenderer->getSwapChain()->Present( 0, 0 );
	}

	void DirectX10ViewRender::setRootNode(Node* a_pRootNode)
	{
		if(a_pRootNode)
			m_pRootNode=a_pRootNode;
	}
}