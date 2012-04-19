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




#include "CudaTextureRender.h"

#include <cuda.h>

#include <builtin_types.h>
#include <cuda_runtime_api.h>

#include <cuda_d3d9_interop.h>
#include <cuda_d3d10_interop.h>
#include <cutil_inline.h>

#include <cassert>
namespace TheWhiteAmbit {
	// The CUDA kernel launchers that get called
	extern "C" 
	{
		bool cuda_texture_2d(void* surface, size_t width, size_t height, size_t pitch, float t);
		//bool cuda_texture_cube(void* surface, int width, int height, size_t pitch, int face, float t);
		//bool cuda_texture_volume(void* surface, int width, int height, int depth, size_t pitch, size_t pitchslice);
	}

	CudaTextureRender::CudaTextureRender(DirectX9Renderer* a_pRenderer)
	{
		m_pRenderer9=a_pRenderer;
		m_pRenderer10=NULL;
		
		m_pDX9Texture2DArray=new DirectX9Texture*[m_iMaxTextures]; //TODO: insert variable texture count
		m_pDX10Texture2DArray=NULL;

		m_pDX9Texture2DArray[0]=NULL;

		if(!g_bCudaD3D9Registered){
			m_cuContext = 0;
			m_cuDevice  = 0;
			//cutilDrvSafeCallNoSync( cuInit(0) );  //TODO: only init device once

			g_bCudaD3D9Registered = true;
			cudaD3D9SetDirect3DDevice(a_pRenderer->getDevice());
			cutilCheckMsg("cudaD3D9SetDirect3DDevice failed");
		}
	}

	CudaTextureRender::CudaTextureRender(DirectX10Renderer* a_pRenderer)
	{
		m_pRenderer9=NULL;
		m_pRenderer10=a_pRenderer;

		m_pDX9Texture2DArray=NULL;
		m_pDX10Texture2DArray=new DirectX10Texture*[m_iMaxTextures]; //TODO: insert variable texture count

		m_pDX10Texture2DArray[0]=NULL;


		if(!g_bCudaD3D10Registered){
			m_cuContext = 0;
			m_cuDevice  = 0;

			g_bCudaD3D10Registered = true;
			cudaD3D10SetDirect3DDevice(a_pRenderer->getDevice());
			cutilCheckMsg("cudaD3D10SetDirect3DDevice failed");
		}
	}

	CudaTextureRender::~CudaTextureRender(void)
	{
		cudaThreadExit();
		cutilCheckMsg("cudaThreadExit failed");
		delete m_pDX9Texture2DArray;
		delete m_pDX10Texture2DArray;
	}

	////////////////////////////////////////////////////////////////////////////////
	//! Run the Cuda part of the computation
	////////////////////////////////////////////////////////////////////////////////
	void CudaTextureRender::RunKernels()
	{
		static float t = 0.0f;

		// populate the 2d texture
		{
			IDirect3DTexture9* tex=this->m_pDX9Texture2DArray[0]->getTexture();
			size_t width=this->m_pDX9Texture2DArray[0]->getWidth();
			size_t height=this->m_pDX9Texture2DArray[0]->getHeight();

			void* pData;
			size_t pitch;
			cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&pData, tex, 0, 0) );
			cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch, NULL, tex, 0, 0) );
			cuda_texture_2d(pData, width, height, pitch, t);
		}

		t += 0.1f;
	}

	void CudaTextureRender::present(int effect)
	{
		if(this->m_pDX9Texture2DArray[0])
		{
			IDirect3DTexture9* tex=this->m_pDX9Texture2DArray[0]->getTexture();

			cudaD3D9MapResources(1, (IDirect3DResource9 **)&tex);
			//cutilSafeCallNoSync ( cudaD3D9MapResources (1, (IDirect3DResource9 **)&tex ));
			cutilCheckMsg("cudaD3D9MapResources(1) failed");


			// run kernels which will populate the contents of the texture
			//
			RunKernels();

			// unmap the resources
			//
			cudaD3D9UnmapResources(1, (IDirect3DResource9 **)&tex);
			cutilCheckMsg("cudaD3D9UnmapResources(1) failed");
		}
		if(this->m_pDX10Texture2DArray[0])
		{
			ID3D10Resource* tex=this->m_pDX10Texture2DArray[0]->getTexture();

			cudaD3D10MapResources(1, (ID3D10Resource **)&tex);
			cutilCheckMsg("cudaD3D9MapResources(1) failed");

			// run kernels which will populate the contents of the texture
			//
			RunKernels();

			// unmap the resources
			//
			cudaD3D10UnmapResources(1, (ID3D10Resource **)&tex);
			cutilCheckMsg("cudaD3D10UnmapResources(1) failed");
		}
	}

	void CudaTextureRender::setTextureTarget(unsigned int a_iTextureNumber, DirectX9Texture* a_pTexture)
	{
		//TODO: evaluate a_iTextureNumber to handle multiple textures
		this->m_pDX9Texture2DArray[a_iTextureNumber]=a_pTexture;
		IDirect3DTexture9* tex=this->m_pDX9Texture2DArray[a_iTextureNumber]->getTexture();
		// register the Direct3D resources that we'll use
		// we'll read to and write from g_texture_2d, so don't set any special map flags for it
		cudaD3D9RegisterResource(tex, cudaD3D9RegisterFlagsNone);
		cutilCheckMsg("cudaD3D9RegisterResource (g_texture_2d) failed");

		// Initialize this texture to be mid grey
		{
			cutilSafeCallNoSync ( cudaD3D9MapResources (1, (IDirect3DResource9 **)&tex ));
			void* data;
			size_t size;
			cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&data, tex, 0, 0) );
			cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedSize(&size, tex, 0, 0) );
			cudaMemset(data, 128, size);
			cutilSafeCallNoSync ( cudaD3D9UnmapResources (1, (IDirect3DResource9 **)&tex ));
		}
		cutilCheckMsg("cudaD3D9UnmapResources (1) failed");
	}

	void CudaTextureRender::setTextureTarget(unsigned int a_iTextureNumber, DirectX10Texture* a_pTexture)
	{
		//TODO: evaluate a_iTextureNumber to handle multiple textures
		this->m_pDX10Texture2DArray[a_iTextureNumber]=a_pTexture;
		ID3D10Resource* tex=this->m_pDX10Texture2DArray[a_iTextureNumber]->getTexture();
		// register the Direct3D resources that we'll use
		// we'll read to and write from g_texture_2d, so don't set any special map flags for it
		cudaD3D10RegisterResource(tex, cudaD3D10RegisterFlagsNone);
		cutilCheckMsg("cudaD3D10RegisterResource (g_texture_2d) failed");

		// Initialize this texture to be mid grey
		{
			cutilSafeCallNoSync ( cudaD3D10MapResources (1, (ID3D10Resource **)&tex ));
			void* data;
			size_t size;
			cutilSafeCallNoSync ( cudaD3D10ResourceGetMappedPointer(&data, tex, 0) );
			cutilSafeCallNoSync ( cudaD3D10ResourceGetMappedSize(&size, tex, 0) );
			cudaMemset(data, 128, size);
			cutilSafeCallNoSync ( cudaD3D10UnmapResources (1, (ID3D10Resource **)&tex ));
		}
		cutilCheckMsg("cudaD3D10UnmapResources (1) failed");
	}
}