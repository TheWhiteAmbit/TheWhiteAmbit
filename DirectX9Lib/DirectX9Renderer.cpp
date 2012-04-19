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



#include "DirectX9Renderer.h"

#pragma comment (lib, "d3d9.lib")
#pragma comment (lib, "d3dx9.lib")

namespace TheWhiteAmbit {

	DirectX9Renderer::DirectX9Renderer(HWND	a_hWindow)
	{
		m_pD3D = NULL;
		m_pd3dDevice = NULL;
		m_hWindow = a_hWindow;

		D3DPRESENT_PARAMETERS d3dpp;
		ZeroMemory( &d3dpp, sizeof( d3dpp ) );
		d3dpp.Windowed = TRUE;
		d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
		d3dpp.BackBufferFormat = D3DFMT_UNKNOWN;
		d3dpp.EnableAutoDepthStencil = TRUE;
		d3dpp.AutoDepthStencilFormat = D3DFMT_D24S8;

		InitDevice(d3dpp);
	}

	DirectX9Renderer::DirectX9Renderer(HWND	a_hWindow, D3DPRESENT_PARAMETERS d3dpp)
	{
		m_pD3D = NULL;
		m_pd3dDevice = NULL;
		m_hWindow = a_hWindow;
		InitDevice(d3dpp);
	}

	DirectX9Renderer::~DirectX9Renderer(void)
	{
		CleanupDevice();
	}


	DWORD DirectX9Renderer::GetVertexProcessingCaps(IDirect3D9* a_pD3D)
	{
		D3DCAPS9 caps;
		DWORD dwVertexProcessing = D3DCREATE_SOFTWARE_VERTEXPROCESSING;
		if (SUCCEEDED(a_pD3D->GetDeviceCaps(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, &caps))){
			if ((caps.DevCaps & D3DDEVCAPS_HWTRANSFORMANDLIGHT) == D3DDEVCAPS_HWTRANSFORMANDLIGHT){
				dwVertexProcessing = D3DCREATE_HARDWARE_VERTEXPROCESSING;
			}
		}
		return dwVertexProcessing;
	}

	HRESULT DirectX9Renderer::InitDevice(D3DPRESENT_PARAMETERS d3dpp)
	{
		// Create the D3D object, which is needed to create the D3DDevice.
		if( NULL == ( m_pD3D = Direct3DCreate9( D3D_SDK_VERSION ) ) )
			return E_FAIL;

		////D3DCREATE_PUREDEVICE
		////D3DCREATE_MULTITHREADED
		//if( FAILED( m_pD3D->CreateDevice( D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, m_hWindow,
		//	D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED,
		//	&d3dpp, &m_pd3dDevice ) ) )
		//{
		//	if( FAILED( m_pD3D->CreateDevice( D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, m_hWindow,
		//		D3DCREATE_SOFTWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED,
		//		&d3dpp, &m_pd3dDevice ) ) )
		//	{
		//		m_pd3dDevice=NULL;
		//		return E_FAIL;
		//	}
		//}

		// determine what type of vertex processing to use based on the device capabilities
		DWORD dwVertexProcessing = GetVertexProcessingCaps(m_pD3D);
		// create the D3D device
		if (FAILED(m_pD3D->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, m_hWindow,
			//dwVertexProcessing | D3DCREATE_MULTITHREADED | D3DCREATE_FPU_PRESERVE,
			dwVertexProcessing | D3DCREATE_MULTITHREADED,
			//dwVertexProcessing, //Very slow
			&d3dpp, &m_pd3dDevice))){
				return E_FAIL;
		}

		m_pd3dDevice->GetRenderTarget(0,&m_pBackBuffer);
		m_pd3dDevice->GetDepthStencilSurface(&m_pZStencilSurface);

		return S_OK;
	}

	IDirect3DSurface9* DirectX9Renderer::getRenderTargetView(void)
	{
		return m_pBackBuffer;
	}

	IDirect3DSurface9* DirectX9Renderer::getDepthStencilView(void)
	{
		return m_pZStencilSurface;
	}

	void DirectX9Renderer::CleanupDevice()
	{
		if( m_pd3dDevice != NULL )
			m_pd3dDevice->Release();

		if( m_pD3D != NULL )
			m_pD3D->Release();
	}

	HWND	DirectX9Renderer::getWindowHandle()
	{
		return this->m_hWindow;
	}

	LPDIRECT3DDEVICE9 DirectX9Renderer::getDevice(void)
	{
		return m_pd3dDevice;
	}
}