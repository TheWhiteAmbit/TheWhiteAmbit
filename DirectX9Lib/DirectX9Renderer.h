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



#pragma once

#define D3DXFX_LARGEADDRESS_HANDLE

#include <d3d9.h>
#include <d3dx9.h>
namespace TheWhiteAmbit {
	class DirectX9Renderer
	{
		IDirect3D9*         m_pD3D; // Used to create the D3DDevice
		IDirect3DDevice9*   m_pd3dDevice; // Our rendering device
		HWND				m_hWindow;
		DWORD				GetVertexProcessingCaps(IDirect3D9* a_pD3D);
		HRESULT				InitDevice(D3DPRESENT_PARAMETERS d3dpp);
		void				CleanupDevice();
		IDirect3DSurface9*	m_pBackBuffer;
		IDirect3DSurface9*	m_pZStencilSurface;
	public:
		IDirect3DDevice9* getDevice(void);
		IDirect3DSurface9* getRenderTargetView(void);
		IDirect3DSurface9* getDepthStencilView(void);

		HWND	getWindowHandle();
		DirectX9Renderer(HWND	a_hWindow);
		DirectX9Renderer(HWND	a_hWindow, D3DPRESENT_PARAMETERS d3dpp);
		virtual ~DirectX9Renderer(void);
	};
}
