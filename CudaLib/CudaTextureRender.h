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

#include "../DirectX9Lib/DirectX9Renderer.h"
#include "../DirectX9Lib/DirectX9Texture.h"

#include "../DirectX10Lib/DirectX10Renderer.h"
#include "../DirectX10Lib/DirectX10Texture.h"

#include <cuda.h>
namespace TheWhiteAmbit {

	static bool g_bCudaD3D9Registered = false;
	static bool g_bCudaD3D10Registered = false;

	class CudaTextureRender
	{
	protected:
		// Data structure for 2D texture shared between DX9 and CUDA
		static const unsigned int	m_iMaxTextures = 8;
		static const unsigned int	m_iNumTargetTextures = 4;
		DirectX9Texture**	m_pDX9Texture2DArray;
		DirectX9Renderer*	m_pRenderer9;

		DirectX10Texture**	m_pDX10Texture2DArray;
		DirectX10Renderer*	m_pRenderer10;

		CUcontext          m_cuContext;
		CUdevice           m_cuDevice;	

		virtual void RunKernels();
	public:
		CudaTextureRender(DirectX9Renderer* a_pRenderer);
		CudaTextureRender(DirectX10Renderer* a_pRenderer);
		virtual ~CudaTextureRender(void);
		virtual void present(int effect);
		virtual void setTextureTarget(unsigned int a_iTextureNumber, DirectX9Texture* a_pTexture);
		virtual void setTextureTarget(unsigned int a_iTextureNumber, DirectX10Texture* a_pTexture);
	};
}