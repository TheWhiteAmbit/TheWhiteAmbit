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

//#define D3D10_IGNORE_SDK_LAYERS

#include <d3d10_1.h>
#include <d3d10.h>
#include <d3dx10.h>
namespace TheWhiteAmbit {
	class DirectX10Renderer
	{
		D3D10_DRIVER_TYPE       m_driverType;
		ID3D10Device*           m_pd3dDevice;
		IDXGISwapChain*         m_pSwapChain;
		ID3D10RenderTargetView* m_pRenderTargetView;
		ID3D10DepthStencilView* m_pDepthStencilView;
		HWND					m_hWindow;
		HRESULT InitDevice();
		void CleanupDevice();
	public:
		HWND	getWindowHandle();
		ID3D10Device* getDevice(void);
		IDXGISwapChain* getSwapChain(void);
		ID3D10RenderTargetView* getRenderTargetView(void);
		ID3D10DepthStencilView* getDepthStencilView(void);
		void activateRenderTarget(void);

		DirectX10Renderer(HWND	a_hWindow);
		virtual ~DirectX10Renderer(void);
	};
}