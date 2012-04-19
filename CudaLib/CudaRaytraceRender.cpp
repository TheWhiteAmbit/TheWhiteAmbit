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




#include "CudaRaytraceRender.h"


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
		bool cuda_bsp_raytrace(void* surfaceDst, void* surfaceDst1, void* surfaceDst2, void* surfaceDst3, void* surfaceSrc, void* surfaceSrc1, void* surfaceSrc2, void* surfaceSrc3, size_t width, size_t height, size_t pitch, void* tris, void* projMatrices, void* splitPlanes, void* splitIndices, int effect);
	}


	CudaRaytraceRender::CudaRaytraceRender(DirectX9Renderer* a_pRenderer) 
		: CudaTextureRender(a_pRenderer)
	{
		this->m_bInitDone=false;
		this->m_pUnitTriangles=NULL;
		this->m_pSplitPlanes=NULL;
		this->m_pSplitIndices=NULL;
		this->m_pProjectionMatrices=NULL;
		this->m_pBSPMesh=NULL;
		this->m_pDX9Texture2DArray=new DirectX9Texture*[m_iMaxTextures]; //TODO: insert variable texture count
		this->m_pDX10Texture2DArray=NULL;
		for (unsigned int i=0; i<m_iMaxTextures; i++)
			this->m_pDX9Texture2DArray[i]=NULL;

		this->m_pRootNode=NULL;
	}

	CudaRaytraceRender::CudaRaytraceRender(DirectX10Renderer* a_pRenderer) 
		: CudaTextureRender(a_pRenderer)
	{
		this->m_bInitDone=false;
		this->m_pUnitTriangles=NULL;
		this->m_pSplitPlanes=NULL;
		this->m_pSplitIndices=NULL;
		this->m_pProjectionMatrices=NULL;
		this->m_pBSPMesh=NULL;
		this->m_pDX9Texture2DArray=NULL;
		this->m_pDX10Texture2DArray=new DirectX10Texture*[m_iMaxTextures]; //TODO: insert variable texture count
		for (unsigned int i=0; i<m_iMaxTextures; i++)
			this->m_pDX10Texture2DArray[i]=NULL;

		this->m_pRootNode=NULL;
	}

	CudaRaytraceRender::~CudaRaytraceRender(void)
	{
		this->m_bInitDone=false;
		if(this->m_pUnitTriangles)
			cudaFree((void*)this->m_pUnitTriangles);
		if(this->m_pProjectionMatrices)
			cudaFree((void*)this->m_pProjectionMatrices);
		if(this->m_pSplitPlanes)
			cudaFree((void*)this->m_pSplitPlanes);
		if(this->m_pSplitIndices)
			cudaFree((void*)this->m_pSplitIndices);

		cudaThreadExit();
		cutilCheckMsg("cudaThreadExit failed");
	}

	void CudaRaytraceRender::setBSPMesh(BSPMesh* a_pBSPMesh)
	{
		if(m_pBSPMesh!=a_pBSPMesh)
		{
			m_pBSPMesh=a_pBSPMesh;
			m_pBSPMesh->generateSplitArrays();

			size_t meshSize=(m_pBSPMesh->getNumSplitsUnitTri()*UNIT_TRIANGLE_MAX)*sizeof(D3DXMATRIX);
			//size_t meshSize=sizeof(D3DXMATRIX);
			if(this->m_pUnitTriangles)
				cudaFree((void*)this->m_pUnitTriangles);
			cudaMalloc((void**)&this->m_pUnitTriangles, meshSize);
			cudaMemcpy((void*)this->m_pUnitTriangles, (void*)m_pBSPMesh->getSplitUnitTriArray(), meshSize, cudaMemcpyHostToDevice);

			//cudaChannelFormatDesc format=cudaCreateChannelDesc(4,4,4,4,cudaChannelFormatKindFloat);
			//cudaMallocArray((cudaArray**)&this->m_pUnitTriangles, &format, meshSize/UNIT_TRIANGLE_MAX, UNIT_TRIANGLE_MAX);
			//cudaMemcpyToArray(d_u,0,0,u, 2050*2050*sizeof(float),cudaMemcpyHostToDevice);
			//cudaBindTextureToArray(uT,d_u);

			size_t planesSize=m_pBSPMesh->getNumSplits()*sizeof(D3DXPLANE);
			if(this->m_pSplitPlanes)
				cudaFree((void*)this->m_pSplitPlanes);
			cudaMalloc((void**)&this->m_pSplitPlanes, planesSize);
			cudaMemcpy((void*)this->m_pSplitPlanes, (void*)m_pBSPMesh->getSplitPlanes(), planesSize, cudaMemcpyHostToDevice);

			size_t indicesSize=4*m_pBSPMesh->getNumSplits()*sizeof(int);
			if(this->m_pSplitIndices)
				cudaFree((void*)this->m_pSplitIndices);
			cudaMalloc((void**)&this->m_pSplitIndices, indicesSize);
			cudaMemcpy((void*)this->m_pSplitIndices, (void*)m_pBSPMesh->getSplitIndices(), indicesSize, cudaMemcpyHostToDevice);

			if(this->m_pProjectionMatrices)
				cudaFree((void*)this->m_pProjectionMatrices);
			cudaMalloc((void**)&this->m_pProjectionMatrices, 24*sizeof(D3DXMATRIX));

			this->m_bInitDone=true; 
		}
	}


	////////////////////////////////////////////////////////////////////////////////
	//! Run the Cuda part of the computation
	////////////////////////////////////////////////////////////////////////////////
	void CudaRaytraceRender::RunKernelsEffect(int effect)
	{
		static float t = 0.0f;
		//TODO: make kernel fit for DX10 also
		// raytrace the 2d texture
		{
			size_t pitch=0;
			size_t width=0;
			size_t height=0;

			//TODO: iterate Texture Array
			void* pSurfaceDestination=NULL;
			if(this->m_pDX9Texture2DArray[0]) {
				IDirect3DTexture9* pTextureDesination=this->m_pDX9Texture2DArray[0]->getTexture();
				width=this->m_pDX9Texture2DArray[0]->getWidth();
				height=this->m_pDX9Texture2DArray[0]->getHeight();
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&pSurfaceDestination, pTextureDesination, 0, 0) );
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch, NULL, pTextureDesination, 0, 0) );
			}
			void* pSurfaceDestination1=NULL;
			if(this->m_pDX9Texture2DArray[1]) {
				IDirect3DTexture9* pTextureDesination1=this->m_pDX9Texture2DArray[1]->getTexture();
				width=this->m_pDX9Texture2DArray[1]->getWidth();
				height=this->m_pDX9Texture2DArray[1]->getHeight();
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&pSurfaceDestination1, pTextureDesination1, 0, 0) );
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch, NULL, pTextureDesination1, 0, 0) );
			}
			void* pSurfaceDestination2=NULL;
			if(this->m_pDX9Texture2DArray[2]) {
				IDirect3DTexture9* pTextureDesination2=this->m_pDX9Texture2DArray[2]->getTexture();
				width=this->m_pDX9Texture2DArray[2]->getWidth();
				height=this->m_pDX9Texture2DArray[2]->getHeight();
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&pSurfaceDestination2, pTextureDesination2, 0, 0) );
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch, NULL, pTextureDesination2, 0, 0) );
			}
			void* pSurfaceDestination3=NULL;
			if(this->m_pDX9Texture2DArray[3]) {
				IDirect3DTexture9* pTextureDesination3=this->m_pDX9Texture2DArray[3]->getTexture();
				width=this->m_pDX9Texture2DArray[3]->getWidth();
				height=this->m_pDX9Texture2DArray[3]->getHeight();
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&pSurfaceDestination3, pTextureDesination3, 0, 0) );
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch, NULL, pTextureDesination3, 0, 0) );
			}
			///////////////
			void* pSurfaceSource=NULL;
			if(this->m_pDX9Texture2DArray[m_iNumTargetTextures]) {
				IDirect3DTexture9* pTextureSource=this->m_pDX9Texture2DArray[m_iNumTargetTextures]->getTexture();
				size_t width0=this->m_pDX9Texture2DArray[m_iNumTargetTextures]->getWidth();
				size_t height0=this->m_pDX9Texture2DArray[m_iNumTargetTextures]->getHeight();
				size_t pitch0;
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&pSurfaceSource, pTextureSource, 0, 0) );
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch0, NULL, pTextureSource, 0, 0) );
			}
			void* pSurfaceSource1=NULL;
			if(this->m_pDX9Texture2DArray[m_iNumTargetTextures+1]) {
				IDirect3DTexture9* pTextureSource1=this->m_pDX9Texture2DArray[m_iNumTargetTextures+1]->getTexture();
				size_t width1=this->m_pDX9Texture2DArray[m_iNumTargetTextures+1]->getWidth();
				size_t height1=this->m_pDX9Texture2DArray[m_iNumTargetTextures+1]->getHeight();
				size_t pitch1;
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&pSurfaceSource1, pTextureSource1, 0, 0) );
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch1, NULL, pTextureSource1, 0, 0) );
			}
			void* pSurfaceSource2=NULL;
			if(this->m_pDX9Texture2DArray[m_iNumTargetTextures+2]) {
				IDirect3DTexture9* pTextureSource2=this->m_pDX9Texture2DArray[m_iNumTargetTextures+2]->getTexture();
				size_t width2=this->m_pDX9Texture2DArray[m_iNumTargetTextures+2]->getWidth();
				size_t height2=this->m_pDX9Texture2DArray[m_iNumTargetTextures+2]->getHeight();
				size_t pitch2;
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&pSurfaceSource2, pTextureSource2, 0, 0) );
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch2, NULL, pTextureSource2, 0, 0) );
			}
			void* pSurfaceSource3=NULL;
			if(this->m_pDX9Texture2DArray[m_iNumTargetTextures+3]) {
				IDirect3DTexture9* pTextureSource3=this->m_pDX9Texture2DArray[m_iNumTargetTextures+3]->getTexture();
				size_t width3=this->m_pDX9Texture2DArray[m_iNumTargetTextures+3]->getWidth();
				size_t height3=this->m_pDX9Texture2DArray[m_iNumTargetTextures+3]->getHeight();
				size_t pitch3;
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&pSurfaceSource3, pTextureSource3, 0, 0) );
				cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch3, NULL, pTextureSource3, 0, 0) );
			}

			cuda_bsp_raytrace(
				pSurfaceDestination,
				pSurfaceDestination1,
				pSurfaceDestination2,
				pSurfaceDestination3,
				pSurfaceSource,
				pSurfaceSource1,
				pSurfaceSource2,
				pSurfaceSource3,
				width, 
				height, 
				pitch,
				(void*)this->m_pUnitTriangles, 
				(void*)this->m_pProjectionMatrices, 
				(void*)this->m_pSplitPlanes, 
				(void*)this->m_pSplitIndices,
				effect);
		}

		t += 0.1f;
	}

	void CudaRaytraceRender::setRootNode(Node* a_pNode)
	{
		m_pRootNode = a_pNode;
	}

	void CudaRaytraceRender::present(int effect)
	{
		if(this->m_pRootNode && this->m_pDX9Texture2DArray) //TODO: insert variable texture count
		{
			RaytraceVisitor raytraceVisitor;
			this->m_pRootNode->accept(&raytraceVisitor);
			this->setBSPMesh(raytraceVisitor.getBSPMesh());

			cudaMemcpy((void*)this->m_pProjectionMatrices, (void*)raytraceVisitor.getMatrices(), 24*sizeof(D3DXMATRIX), cudaMemcpyHostToDevice);

			IDirect3DResource9* ppResources[this->m_iMaxTextures] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };

			unsigned int resourceCount=0;
			for (unsigned int i=0; i<m_iMaxTextures; i++) {
				if(this->m_pDX9Texture2DArray[i]) {
					ppResources[resourceCount]=this->m_pDX9Texture2DArray[i]->getTexture();
					resourceCount++;
				}
			}

			cudaThreadSynchronize();		

			cudaD3D9MapResources(resourceCount, (IDirect3DResource9**)ppResources);
			cutilCheckMsg("cudaD3D9MapResources(resourceCount) failed");

			// run kernels which will populate the contents of the texture
			//
			RunKernelsEffect(effect);

			// unmap the resources
			//
			cudaD3D9UnmapResources(resourceCount, ppResources);
			cutilCheckMsg("cudaD3D9UnmapResources(resourceCount) failed");
		}
		if(this->m_pRootNode && this->m_pDX10Texture2DArray) //TODO: insert variable texture count
		{
			//TODO: make direct3d10 handle resources more flexible as with directx9 above

			RaytraceVisitor raytraceVisitor;
			this->m_pRootNode->accept(&raytraceVisitor);
			this->setBSPMesh(raytraceVisitor.getBSPMesh());

			cudaMemcpy((void*)this->m_pProjectionMatrices, (void*)raytraceVisitor.getMatrices(), 24*sizeof(D3DXMATRIX), cudaMemcpyHostToDevice);


			ID3D10Resource* ppResources[this->m_iMaxTextures];

			unsigned int resourceCount=0;
			for (unsigned int i=0; i<m_iMaxTextures; i++) {
				if(this->m_pDX10Texture2DArray[i]) {
					ppResources[resourceCount]=this->m_pDX10Texture2DArray[i]->getTexture();
					resourceCount++;
				}
			}

			cudaThreadSynchronize();
			cudaD3D10MapResources(resourceCount, ppResources);
			cutilCheckMsg("cudaD3D10MapResources(resourceCount) failed");

			// run kernels which will populate the contents of the texture
			//
			RunKernelsEffect(effect);

			// unmap the resources
			//
			cudaD3D10UnmapResources(resourceCount, ppResources);
			cutilCheckMsg("cudaD3D10UnmapResources(resourceCount) failed");
		}
	}


	//void CudaRaytraceRender::setTextureSource(unsigned int a_iTextureNumber, DirectX9Texture* a_pTexture)
	//{
	//	this->m_pDX9Texture2DArray[m_iNumTargetTextures+a_iTextureNumber]=a_pTexture;
	//	IDirect3DTexture9* tex=this->m_pDX9Texture2DArray[m_iNumTargetTextures+a_iTextureNumber]->getTexture();
	//	// register the Direct3D resources that we'll use
	//	// we'll read to and write from g_texture_2d, so don't set any special map flags for it
	//	cudaD3D9RegisterResource(tex, cudaD3D9RegisterFlagsNone);
	//	cutilCheckMsg("cudaD3D9RegisterResource (g_texture_2d) failed");

	//	// Initialize this texture to be mid grey
	//	{
	//		cutilSafeCallNoSync ( cudaD3D9MapResources (1, (IDirect3DResource9 **)&tex ));
	//		void* data;
	//		size_t size;
	//		cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&data, tex, 0, 0) );
	//		cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedSize(&size, tex, 0, 0) );
	//		cudaMemset(data, 128, size);
	//		cutilSafeCallNoSync ( cudaD3D9UnmapResources (1, (IDirect3DResource9 **)&tex ));
	//	}
	//	cutilCheckMsg("cudaD3D9UnmapResources (1) failed");
	//}

	//void CudaRaytraceRender::setTextureSource(unsigned int a_iTextureNumber, DirectX10Texture* a_pTexture)
	//{
	//	//TODO: implement multitexture handling for direct3d10

	//	this->m_pDX10Texture2DArray[m_iNumTargetTextures+a_iTextureNumber]=a_pTexture;
	//	ID3D10Resource* tex=this->m_pDX10Texture2DArray[m_iNumTargetTextures+a_iTextureNumber]->getTexture();
	//	// register the Direct3D resources that we'll use
	//	// we'll read to and write from g_texture_2d, so don't set any special map flags for it
	//	cudaD3D10RegisterResource(tex, cudaD3D10RegisterFlagsNone);
	//	cutilCheckMsg("cudaD3D10RegisterResource (g_texture_2d) failed");

	//	// Initialize this texture to be mid grey
	//	{
	//		cutilSafeCallNoSync ( cudaD3D10MapResources (1, (ID3D10Resource **)&tex ));
	//		void* data;
	//		size_t size;
	//		cutilSafeCallNoSync ( cudaD3D10ResourceGetMappedPointer(&data, tex, 0) );
	//		cutilSafeCallNoSync ( cudaD3D10ResourceGetMappedSize(&size, tex, 0) );
	//		cudaMemset(data, 128, size);
	//		cutilSafeCallNoSync ( cudaD3D10UnmapResources (1, (ID3D10Resource **)&tex ));
	//	}
	//	cutilCheckMsg("cudaD3D10UnmapResources (1) failed");
	//}
}