#pragma once
#include "CudaTextureRender.h"
#include "VideoEncoder.h"

//static unsigned char* _stdcall HandleAcquireBitStream(int *pBufferSize, void *pUserData);
//static void _stdcall HandleReleaseBitStream(int nBytesInBuffer, unsigned char *cb,void *pUserData);
//static void _stdcall HandleOnBeginFrame(const NVVE_BeginFrameInfo *pbfi, void *pUserData);
//static void _stdcall HandleOnEndFrame(const NVVE_EndFrameInfo *pefi, void *pUserData);
namespace TheWhiteAmbit {
	class CudaFileRender :
		public CudaTextureRender
	{
		DirectX9Texture*	m_pDX9Texture2DSourceRGBA;
		DirectX10Texture*	m_pDX10Texture2DSourceRGBA;

		//DirectX9Texture*	m_pDX9Texture2DSourceNV12;
		//DirectX10Texture*	m_pDX10Texture2DSourceNV12;

		IDirect3DTexture9*	m_ppRegisteredResources[2];

		bool                m_bIsProgressive;

		CUdeviceptr        m_pFrameBuffer;
		unsigned int        m_iFrameBufferPitch;

		// from cuda encoder

		// NVCUVENC data structures and wrapper class
		VideoEncoder*	    m_pCudaEncoder;
		NVEncoderParams     m_sEncoderParams;
		NVVE_CallbackParams m_sCBParams;

		// CUDA resources needed (for CUDA Encoder interop with a previously created CUDA Context, and accepting GPU video memory)
		//CUcontext      m_cuContext;
		//CUdevice       m_cuDevice;
		CUvideoctxlock m_cuCtxLock;
		CUdeviceptr    m_dptrVideoFrame;

		size_t m_iPitch;
		unsigned int m_iHeight;
		unsigned int m_iWidthInBytes;
		unsigned int m_iElementSizeBytes;

		bool ParseInputParams(NVEncoderParams *pParams, LPCWSTR a_pFilename);

		// method declarations from ImageDX
		void registerResources(unsigned int nFrames);
		void map(void** ppImageData, unsigned int * pImagePitch, int active_field = 0);    
		void unmap(int active_field = 0);

	public:
		CudaFileRender(DirectX9Renderer* a_pRenderer, LPCWSTR a_pFilename);
		CudaFileRender(DirectX10Renderer* a_pRenderer, LPCWSTR a_pFilename);

		virtual void present(int effect);

		virtual void setTextureSource(unsigned int a_iTextureNumber, DirectX9Texture* a_pTexture);
		virtual void setTextureSource(unsigned int a_iTextureNumber, DirectX10Texture* a_pTexture);

		virtual ~CudaFileRender(void);
	};
}