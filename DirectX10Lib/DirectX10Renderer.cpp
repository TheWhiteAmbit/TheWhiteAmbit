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



#include "DirectX10Renderer.h"

#pragma comment (lib, "d3d10.lib")
#pragma comment (lib, "d3dx10.lib")

namespace TheWhiteAmbit {
	DirectX10Renderer::DirectX10Renderer(HWND	a_hWindow)
	{
		m_hWindow = a_hWindow;
		m_driverType = D3D10_DRIVER_TYPE_NULL;
		m_pd3dDevice = NULL;
		m_pSwapChain = NULL;
		m_pRenderTargetView = NULL;
		InitDevice();
	}

	DirectX10Renderer::~DirectX10Renderer(void)
	{
		CleanupDevice();
	}

	//--------------------------------------------------------------------------------------
	// Create Direct3D device and swap chain
	//--------------------------------------------------------------------------------------
	HRESULT DirectX10Renderer::InitDevice()
	{
		HRESULT hr = S_OK;;

		RECT rc;
		GetClientRect( m_hWindow, &rc );
		UINT width = rc.right - rc.left;
		UINT height = rc.bottom - rc.top;

		UINT createDeviceFlags = 0;
#ifdef _DEBUG
		createDeviceFlags |= D3D10_CREATE_DEVICE_DEBUG;
#endif

		D3D10_DRIVER_TYPE driverTypes[] =
		{
			D3D10_DRIVER_TYPE_HARDWARE,
			D3D10_DRIVER_TYPE_REFERENCE,
		};
		UINT numDriverTypes = sizeof( driverTypes ) / sizeof( driverTypes[0] );

		DXGI_SWAP_CHAIN_DESC sd;
		ZeroMemory( &sd, sizeof( sd ) );
		sd.BufferCount = 1;
		sd.BufferDesc.Width = width;
		sd.BufferDesc.Height = height;
		sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
		//sd.BufferDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		sd.BufferDesc.RefreshRate.Numerator = 100;
		sd.BufferDesc.RefreshRate.Denominator = 1;
		sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		sd.OutputWindow = m_hWindow;
		sd.SampleDesc.Count = 1;
		sd.SampleDesc.Quality = 0;
		sd.Windowed = TRUE;

		for( UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++ )
		{
			m_driverType = driverTypes[driverTypeIndex];
			hr = D3D10CreateDeviceAndSwapChain( NULL, m_driverType, NULL, createDeviceFlags,
				D3D10_SDK_VERSION, &sd, &m_pSwapChain, &m_pd3dDevice );
			if( SUCCEEDED( hr ) )
				break;
		}
		if( FAILED( hr ) )
			return hr;

		// Create a render target view
		ID3D10Texture2D* pBackBuffer;
		hr = m_pSwapChain->GetBuffer( 0, __uuidof( ID3D10Texture2D ), ( LPVOID* )&pBackBuffer );
		if( FAILED( hr ) )
			return hr;

		hr = m_pd3dDevice->CreateRenderTargetView( pBackBuffer, NULL, &m_pRenderTargetView );
		pBackBuffer->Release();
		if( FAILED( hr ) )
			return hr;

		//// Create a new Depth-Stencil texture to replace the DXUT created one
		ID3D10Texture2D* pDepthStencilTexture;
		D3D10_TEXTURE2D_DESC descDepth;
		descDepth.Width = width;
		descDepth.Height = height;
		descDepth.MipLevels = 1;
		descDepth.ArraySize = 1;
		descDepth.Format = DXGI_FORMAT_D32_FLOAT;
		descDepth.SampleDesc.Count = 1;
		descDepth.SampleDesc.Quality = 0;
		descDepth.Usage = D3D10_USAGE_DEFAULT;
		descDepth.BindFlags = D3D10_BIND_DEPTH_STENCIL;
		descDepth.CPUAccessFlags = 0;
		descDepth.MiscFlags = 0;
		hr = m_pd3dDevice->CreateTexture2D( &descDepth, NULL, &pDepthStencilTexture );
		if( FAILED(hr) )
			return hr;

		D3D10_DEPTH_STENCIL_VIEW_DESC descDSV;
		descDSV.Format = descDepth.Format;
		descDSV.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2D;
		descDSV.Texture2D.MipSlice = 0;

		hr = m_pd3dDevice->CreateDepthStencilView(pDepthStencilTexture, &descDSV, &m_pDepthStencilView);
		//pDepthStencilTexture->Release();
		if( FAILED( hr ) )
			return hr;

		m_pd3dDevice->OMSetRenderTargets( 1, &m_pRenderTargetView, m_pDepthStencilView );
		//m_pd3dDevice->OMSetRenderTargets( 1, &m_pRenderTargetView, NULL );

		// Setup the viewport
		D3D10_VIEWPORT vp;
		vp.Width = width;
		vp.Height = height;
		vp.MinDepth = 0.0f;
		vp.MaxDepth = 1.0f;
		vp.TopLeftX = 0;
		vp.TopLeftY = 0;
		m_pd3dDevice->RSSetViewports( 1, &vp );

		return S_OK;
	}

	//--------------------------------------------------------------------------------------
	// Clean up the objects we've created
	//--------------------------------------------------------------------------------------
	void DirectX10Renderer::CleanupDevice()
	{
		if( m_pd3dDevice ) m_pd3dDevice->ClearState();

		if( m_pRenderTargetView ) m_pRenderTargetView->Release();
		if( m_pDepthStencilView ) m_pDepthStencilView->Release();
		if( m_pSwapChain ) m_pSwapChain->Release();
		if( m_pd3dDevice ) m_pd3dDevice->Release();
	}

	void DirectX10Renderer::activateRenderTarget(void)
	{
		if( m_pd3dDevice )
			m_pd3dDevice->OMSetRenderTargets( 1, &m_pRenderTargetView, m_pDepthStencilView );
	}	

	HWND	DirectX10Renderer::getWindowHandle()
	{
		return this->m_hWindow;
	}

	ID3D10Device* DirectX10Renderer::getDevice(void)
	{
		while(!m_pd3dDevice)
			Sleep(10);
		return m_pd3dDevice;
	}

	IDXGISwapChain* DirectX10Renderer::getSwapChain(void)
	{
		return m_pSwapChain;
	}

	ID3D10RenderTargetView* DirectX10Renderer::getRenderTargetView(void)
	{
		return m_pRenderTargetView;
	}

	ID3D10DepthStencilView* DirectX10Renderer::getDepthStencilView(void)
	{
		return m_pDepthStencilView;
	}
}