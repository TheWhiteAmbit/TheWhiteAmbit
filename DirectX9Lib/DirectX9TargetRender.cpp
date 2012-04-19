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




#include "DirectX9TargetRender.h"
#include "../SceneGraphLib/RenderVisitor.h"
#include "../SceneGraphLib/TransformVisitor.h"
namespace TheWhiteAmbit {
	DirectX9TargetRender::DirectX9TargetRender(DirectX9Renderer* a_pRenderer)
	{
		m_pRenderer=a_pRenderer;
		m_iTicks=0;
		m_pRootNode = NULL;

		// Create the render target texture
		m_iWidth=(unsigned int)RESOLUTION_X;
		m_iHeight=(unsigned int)RESOLUTION_Y;

		m_pRenderTexture=new DirectX9Texture(m_pRenderer, m_iWidth, m_iHeight, D3DUSAGE_RENDERTARGET, D3DFMT_A32B32G32R32F);

		m_pRenderer->getDevice()->CreateDepthStencilSurface( 
			m_iWidth,
			m_iHeight,
			//D3DFMT_D32F_LOCKABLE, 
			//causes wpf renderer to have no depthtest
			//http://msdn.microsoft.com/en-us/library/bb172558(VS.85).aspx
			//only in Direct3D9Ex

			D3DFMT_D24S8,
			D3DMULTISAMPLE_NONE,
			0,
			FALSE,
			&m_pDepthSurface,
			NULL);
	}


	DirectX9TargetRender::DirectX9TargetRender(DirectX9Renderer* a_pRenderer, D3DFORMAT a_D3dFormat)
	{
		m_pRenderer=a_pRenderer;
		m_iTicks=0;
		m_pRootNode = NULL;

		// Create the render target texture
		m_iWidth=(unsigned int)RESOLUTION_X;
		m_iHeight=(unsigned int)RESOLUTION_Y;

		m_pRenderTexture=new DirectX9Texture(m_pRenderer, m_iWidth, m_iHeight, D3DUSAGE_RENDERTARGET, a_D3dFormat);

		HRESULT hr=m_pRenderer->getDevice()->CreateDepthStencilSurface( 
			m_iWidth,
			m_iHeight,
			D3DFMT_D24S8,
			D3DMULTISAMPLE_NONE,
			0,
			FALSE,
			&m_pDepthSurface,
			NULL);
	}

	DirectX9TargetRender::~DirectX9TargetRender(void)
	{
		delete m_pRenderTexture;
		if(m_pDepthSurface)
		{
			m_pDepthSurface->Release();
			m_pDepthSurface=NULL;
		}
	}

	void DirectX9TargetRender::present(IEffect* a_pEffect)
	{
		m_iTicks++;

		m_pRenderer->getDevice()->SetRenderTarget(0, m_pRenderTexture->getSurface(0));
		m_pRenderer->getDevice()->SetDepthStencilSurface(m_pDepthSurface);


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

	void DirectX9TargetRender::setRootNode(Node* a_pNode)
	{
		m_pRootNode = a_pNode;
	}

	DirectX9Texture*	DirectX9TargetRender::getTexture(void)
	{
		//return this->m_pDepthSurface;
		return this->m_pRenderTexture;
	}
}