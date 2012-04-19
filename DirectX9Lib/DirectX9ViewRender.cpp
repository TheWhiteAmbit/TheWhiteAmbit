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



#include "DirectX9ViewRender.h"
#include "../SceneGraphLib/RenderVisitor.h"
#include "../SceneGraphLib/TransformVisitor.h"
#include "../SceneGraphLib/PickVisitor.h"
namespace TheWhiteAmbit {
	DirectX9ViewRender::DirectX9ViewRender(DirectX9Renderer* a_pRenderer)
	{
		m_bRendererInitialized=false;
		m_pRootNode=NULL;
		m_iTicks=0;

		m_pRenderer=a_pRenderer;
	}

	DirectX9ViewRender::~DirectX9ViewRender(void)
	{
	}


	void DirectX9ViewRender::present(IEffect* a_pEffect)
	{
		m_iTicks++;

		m_pRenderer->getDevice()->SetRenderTarget(0,m_pRenderer->getRenderTargetView());
		m_pRenderer->getDevice()->SetDepthStencilSurface(m_pRenderer->getDepthStencilView());

		if( SUCCEEDED( m_pRenderer->getDevice()->BeginScene() ) )
		{
			// Rendering of scene objects can happen here
			RenderVisitor renderNodeVisitor(a_pEffect);
			if(m_pRootNode)
				m_pRootNode->accept(&renderNodeVisitor);

			// End the scene
			m_pRenderer->getDevice()->EndScene();
			m_pRenderer->getDevice()->Present( NULL, NULL, NULL, NULL );
		}
	}

	void DirectX9ViewRender::setRootNode(Node* a_pNode)
	{
		if(a_pNode)
			m_pRootNode=a_pNode;
	}
}
