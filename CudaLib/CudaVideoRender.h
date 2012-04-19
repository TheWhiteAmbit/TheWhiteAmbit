#pragma once
#include "CudaTextureRender.h"

//#ifndef CUDA_FORCE_API_VERSION 
//#define CUDA_FORCE_API_VERSION 3010
//#endif

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
//#include <windows.h>
//#include <windowsx.h>
#endif
//#include <d3dx9.h>

#include <math.h>
#include <cuda.h>

#include "FrameQueue.h"
#include "VideoSource.h"
#include "VideoParser.h"
#include "VideoDecoder.h"

#include "cudaProcessFrame.h"
#include "cudaModuleMgr.h"

#include <memory>
#include <iostream>
#include <cassert>

#ifdef _DEBUG
#define ENABLE_DEBUG_OUT    0
#else
#define ENABLE_DEBUG_OUT    0
#endif

#define STRCASECMP  stricmp
#define STRNCASECMP strnicmp
namespace TheWhiteAmbit {
	class CudaVideoRender :
		public CudaTextureRender
	{
		bool                m_bDone;
		bool                m_bAutoQuit;
		bool                m_bUseVsync;
		bool                m_bQAReadback;
		bool                m_bFirstFrame;
		bool                m_bLoop;
		bool                m_bUpdateCSC;
		bool                m_bUpdateAll;
		bool                m_bInterop;
		bool                m_bReadback; // this flag enables/disables reading back of a video from a window
		bool                m_bIsProgressive ; // assume it is progressive, unless otherwise noted

		cudaVideoCreateFlags m_eVideoCreateFlags ;
		CUvideoctxlock       m_CtxLock ;

		//D3DDISPLAYMODE        m_d3ddm;    
		//D3DPRESENT_PARAMETERS m_d3dpp;    

		// These are CUDA function pointers to the CUDA kernels
		CUmoduleManager  * m_pCudaModule;

		CUmodule           m_cuModNV12toARGB ;
		CUfunction         m_fpNV12toARGB  ;
		CUfunction         m_fpPassthru  ;

		CUstream           m_ReadbackSID;
		CUstream           m_KernelSID;

		eColorSpace        m_eColorSpace ;
		float              m_nHue  ;

		// System Memory surface we want to readback to
		BYTE         * m_bFrameData ;
		FrameQueue   * m_pFrameQueue ;
		VideoSource  * m_pVideoSource ;
		VideoParser  * m_pVideoParser ;
		VideoDecoder * m_pVideoDecoder;

		//ImageDX      * m_pImageDX ;
		CUdeviceptr  * m_pInteropFrame ; // if we're using CUDA malloc

		//std::string sFileName;
		LPCWSTR m_sFileName;

		unsigned int m_nWindowWidth;
		unsigned int m_nWindowHeight;
		unsigned int m_nVideoWidth;
		unsigned int m_nVideoHeight;
		unsigned int m_nFrameCount;
		unsigned int m_nDecodeFrameCount;

		// method declarations from videoDecodeD3D9
		bool loadVideoSource(const wchar_t * video_file, 
			unsigned int & width, unsigned int & height, 
			unsigned int & dispWidth, unsigned int & dispHeight);
		void initCudaVideo( );
		void terminateCudaVideo(bool bDestroyContext);
		bool copyDecodedFrameToTexture(unsigned int &nRepeats, int bUseInterop, int *pbIsProgressive);
		void cudaPostProcessFrame(CUdeviceptr *ppDecodedFrame, unsigned int nDecodedPitch, 
			CUdeviceptr *ppTextureData,  unsigned int nTexturePitch,
			CUmodule m_cuModNV12toARGB, 
			CUfunction fpCudaKernel, CUstream streamID);
		HRESULT cleanup(bool bDestroyContext);
		HRESULT initCudaResources(int bUseInterop, int bTCC);
		void renderVideoFrame( bool bUseInterop );
		HRESULT reinitCudaResources();

		// method declarations from ImageDX
		void registerResources(unsigned int nFrames);
		void map(CUdeviceptr * ppImageData, unsigned int * pImagePitch, int active_field = 0);    
		void unmap(int active_field = 0);

		//own fields
		DirectX9Texture*	m_pDX9Texture2DField1;
		DirectX9Texture*	m_pDX9Texture2DField2;
		IDirect3DTexture9*	m_ppRegisteredResources[2];

		void setVideoFile(LPCWSTR a_pVideoFilename);
	public:	
		CudaVideoRender(DirectX9Renderer* a_pRenderer, LPCWSTR a_pFilename);
		CudaVideoRender(DirectX10Renderer* a_pRenderer, LPCWSTR a_pFilename);

		virtual void present(int effect);
		virtual void setTextureTarget(unsigned int a_iTextureNumber, DirectX9Texture* a_pTexture);
		virtual DirectX9Texture* getTextureTarget(unsigned int a_iTextureNumber);

		virtual unsigned int getVideoWidth();
		virtual unsigned int getVideoHeight();

		virtual void release();
		virtual bool hasNext();

		virtual ~CudaVideoRender(void);
	};
}