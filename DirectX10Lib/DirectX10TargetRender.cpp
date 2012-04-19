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



#include "DirectX10TargetRender.h"
#include "../SceneGraphLib/RenderVisitor.h"
#include "../SceneGraphLib/TransformVisitor.h"
namespace TheWhiteAmbit {
	DirectX10TargetRender::DirectX10TargetRender(DirectX10Renderer* a_pRenderer)
	{
		m_pRenderer=a_pRenderer;
		m_iTicks=0;
		m_pRootNode = NULL;

		// Create the render target texture
		m_iWidth=(unsigned int)RESOLUTION_X;
		m_iHeight=(unsigned int)RESOLUTION_Y;


		m_pRenderTargetTexture = new DirectX10Texture(
			m_pRenderer, m_iWidth, m_iHeight, 
			D3D10_USAGE_DEFAULT, 
			DXGI_FORMAT_R32G32B32A32_FLOAT, 
			D3D10_BIND_RENDER_TARGET | D3D10_BIND_SHADER_RESOURCE); //

		// Create the Render-Target
		HRESULT hr = m_pRenderer->getDevice()->CreateRenderTargetView( m_pRenderTargetTexture->getTexture(), NULL, &m_pRenderTargetView );
		if( FAILED( hr ) )
		{
			delete m_pRenderTargetTexture;
			m_pRenderTargetTexture=NULL;
			return;
		}

		//// Create a new Depth-Stencil texture to replace the DXUT created one
		DXGI_FORMAT depthStencilFormat=DXGI_FORMAT_D24_UNORM_S8_UINT;
		//DXGI_FORMAT_D32_FLOAT;
		m_pDepthStencilTexture = new DirectX10Texture(
			m_pRenderer, m_iWidth, m_iHeight, 
			D3D10_USAGE_DEFAULT, 
			depthStencilFormat, 
			D3D10_BIND_DEPTH_STENCIL);

		// Create the Depth-Stencil
		D3D10_DEPTH_STENCIL_VIEW_DESC descDSV;
		descDSV.Format = depthStencilFormat;
		descDSV.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2D;
		descDSV.Texture2D.MipSlice = 0;
		hr = m_pRenderer->getDevice()->CreateDepthStencilView(m_pDepthStencilTexture->getTexture(), &descDSV, &m_pDepthStencilView);
		//pDepthStencilTexture->Release();
		if( FAILED( hr ) )
			return;

		m_pRenderer->getDevice()->OMSetRenderTargets( 1, &m_pRenderTargetView, m_pDepthStencilView );
		//m_pRenderer->getDevice()->OMSetRenderTargets( 1, &m_pRenderTargetView, NULL );

		//// Setup the viewport
		//D3D10_VIEWPORT vp;
		//vp.Width = width;
		//vp.Height = height;
		//vp.MinDepth = 0.0f;
		//vp.MaxDepth = 1.0f;
		//vp.TopLeftX = 0;
		//vp.TopLeftY = 0;
		//m_pRenderer->getDevice()->RSSetViewports( 1, &vp );
	}

	DirectX10TargetRender::~DirectX10TargetRender(void)
	{
	}

	void DirectX10TargetRender::present(IEffect* a_pEffect)
	{
		m_iTicks++;

		m_pRenderer->getDevice()->OMSetRenderTargets( 1, &m_pRenderTargetView, m_pDepthStencilView );
		RenderVisitor renderNodeVisitor(a_pEffect);
		if(m_pRootNode)
			m_pRootNode->accept(&renderNodeVisitor);

		m_pRenderer->getSwapChain()->Present( 0, 0 );
	}

	void DirectX10TargetRender::setRootNode(Node* a_pNode)
	{
		m_pRootNode = a_pNode;
	}

	DirectX10Texture* DirectX10TargetRender::getTexture(void)
	{
		return this->m_pRenderTargetTexture;
	}
}