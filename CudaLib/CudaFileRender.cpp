
#include "CudaFileRender.h"

#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>

#include <cuda_d3d9_interop.h>
#include <cuda_d3d10_interop.h>
#include <cutil_inline.h>

#include <builtin_types.h>
#include <cudad3d9.h>

#include <cassert>

namespace TheWhiteAmbit {

#define VIDEO_CONFIG_FILE L"720p-h264.cfg"

	// The CUDA kernel launchers that get called
	extern "C" 
	{
		void cuda_yuv_transform(void* surfaceSrc, void* surfaceDst, size_t width, size_t height, size_t pitchSrc, size_t pitchDst);
	}


	// NVCUVENC callback function to signal the start of bitstream that is to be encoded
	static unsigned char* _stdcall HandleAcquireBitStream(int *pBufferSize, void *pUserData)
	{
		VideoEncoder *pCudaEncoder;
		if (pUserData) {
			pCudaEncoder = (VideoEncoder *)pUserData;
		} else {
			printf(">> VideoEncoder structure is invalid!\n");
		}

		*pBufferSize = 1024*1024;
		return pCudaEncoder->GetCharBuf();
	}

	//NVCUVENC callback function to signal that the encoded bitstream is ready to be written to file
	static void _stdcall HandleReleaseBitStream(int nBytesInBuffer, unsigned char *cb,void *pUserData)
	{
		VideoEncoder *pCudaEncoder;
		if (pUserData) {
			pCudaEncoder = (VideoEncoder *)pUserData;
		} else {
			printf(">> VideoEncoder structure is invalid!\n");
			return;
		}

		if ( pCudaEncoder && pCudaEncoder->fileOut() )
			fwrite( cb,1,nBytesInBuffer,pCudaEncoder->fileOut() );
		return;
	}

	//NVCUVENC callback function to signal that the encoding operation on the frame has started
	static void _stdcall HandleOnBeginFrame(const NVVE_BeginFrameInfo *pbfi, void *pUserData)
	{
		return;
	}

	//NVCUVENC callback function signals that the encoding operation on the frame has finished
	static void _stdcall HandleOnEndFrame(const NVVE_EndFrameInfo *pefi, void *pUserData)
	{
		VideoEncoder *pCudaEncoder;
		if (pUserData) {
			pCudaEncoder = (VideoEncoder *)pUserData;
		} else {
			printf(">> VideoEncoder structure is invalid!\n");
			return;
		}

		pCudaEncoder->frameSummation(-(pefi->nFrameNumber));

		if ( pCudaEncoder->IsLastFrameSent()) {
			// Check to see if the last frame has been sent
			if ( pCudaEncoder->getFrameSum() == 0) {
				printf(">> Encoder has finished encoding last frame\n<< n");
			}
		} else {
#ifdef _DEBUG
			//        printf("HandleOnEndFrame (%d), FrameCount (%d), FrameSummation (%d)\n", 
			//			pefi->nFrameNumber, pCudaEncoder->frameCount(), 
			//			pCudaEncoder->getFrameSum());
#endif
		}
		return;
	}

	CudaFileRender::CudaFileRender(DirectX9Renderer* a_pRenderer, LPCWSTR a_pFilename)
		: CudaTextureRender(a_pRenderer)
	{
		HRESULT hr = S_OK;

		m_bIsProgressive = true;
		m_pDX9Texture2DSourceRGBA = NULL;
		m_ppRegisteredResources[0] = NULL;
		m_ppRegisteredResources[1] = NULL;
		m_pFrameBuffer=NULL;
		m_iFrameBufferPitch=0;

		m_pCudaEncoder  = NULL;
		ZeroMemory(&m_sEncoderParams, sizeof(NVEncoderParams));
		ZeroMemory(&m_sCBParams, sizeof(NVVE_CallbackParams));

		// CUDA resources needed (for CUDA Encoder interop with a previously created CUDA Context, and accepting GPU video memory)
		m_cuCtxLock      = 0;
		m_dptrVideoFrame = 0;

		// First we parse the input file (based on the command line parameters)
		// Set the input/output filenmaes
		if (!ParseInputParams(&m_sEncoderParams, a_pFilename)) {
			return;
		}

		// Create the NVCUVENC wrapper class for handling encoding
		m_pCudaEncoder = new VideoEncoder  (&m_sEncoderParams, true);
		m_pCudaEncoder->InitEncoder        (&m_sEncoderParams);
		m_pCudaEncoder->SetEncodeParameters(&m_sEncoderParams); 

		// This is for GPU device memory input, and support for interop with another CUDA context
		// The NVIDIA CUDA Encoder will use this CUDA context to be able to pass in shared device memory
		if (m_sEncoderParams.iUseDeviceMem) {
			HRESULT hr = S_OK;
			printf(">> Using Device Memory for Video Input to CUDA Encoder << \n");

			// Create the CUDA context
			cutilDrvSafeCallNoSync( cuInit(0) );  //TODO: only init device once
			cutilDrvSafeCallNoSync( cuDeviceGet(&m_cuDevice, m_sEncoderParams.iForcedGPU) );
			cutilDrvSafeCallNoSync( cuCtxCreate(&m_cuContext, CU_CTX_BLOCKING_SYNC, m_cuDevice) );

			// Allocate the CUDA memory Pitched Surface
			if (m_sEncoderParams.iSurfaceFormat == UYVY || 
				m_sEncoderParams.iSurfaceFormat == YUY2)
			{
				m_iWidthInBytes     =(m_sEncoderParams.iInputSize[0] * sSurfaceFormat[m_sEncoderParams.iSurfaceFormat].bpp) >> 3; // Width
				m_iHeight           = m_sEncoderParams.iInputSize[1];
			} else {
				m_iWidthInBytes     = m_sEncoderParams.iInputSize[0]; // Width
				m_iHeight           = (unsigned int)(m_sEncoderParams.iInputSize[1] * sSurfaceFormat[m_sEncoderParams.iSurfaceFormat].bpp) >> 3;
			}

			m_iElementSizeBytes = 16;
			cutilDrvSafeCallNoSync( cuMemAllocPitch( &m_dptrVideoFrame, &m_iPitch, m_iWidthInBytes, m_iHeight, m_iElementSizeBytes ) );
			m_sEncoderParams.nDeviceMemPitch = m_iPitch; // Copy the Device Memory Pitch (we'll need this for later if we use device memory)

			// Pop the CUDA context from the stack (this will make the CUDA context current)
			// This is needed in order to inherit the CUDA contexts created outside of the CUDA H.264 Encoder
			// CUDA H.264 Encoder will just inherit the available CUDA context
			CUcontext cuContextCurr;
			cutilDrvSafeCallNoSync( cuCtxPopCurrent(&cuContextCurr) );

			// Create the Video Context Lock (used for synchronization)
			cutilDrvSafeCallNoSync( cuvidCtxLockCreate(&m_cuCtxLock, m_cuContext) );

			// If we are using GPU Device Memory with NVCUVENC, it is necessary to create a 
			// CUDA Context with a Context Lock cuvidCtxLock.  The Context Lock needs to be passed to NVCUVENC
			{
				hr = m_pCudaEncoder->SetParamValue(NVVE_DEVICE_MEMORY_INPUT, &(m_sEncoderParams.iUseDeviceMem));
				if (FAILED(hr)) {
					printf("NVVE_DEVICE_MEMORY_INPUT failed\n");
				}
				hr = m_pCudaEncoder->SetParamValue(NVVE_DEVICE_CTX_LOCK    , &m_cuCtxLock);
				if (FAILED(hr)) {
					printf("NVVE_DEVICE_CTX_LOCK failed\n");
				}
			}
		}

		// Now provide the callback functions to CUDA H.264 Encoder
		{
			memset(&m_sCBParams,0,sizeof(NVVE_CallbackParams));
			m_sCBParams.pfnacquirebitstream = HandleAcquireBitStream;
			m_sCBParams.pfnonbeginframe     = HandleOnBeginFrame;
			m_sCBParams.pfnonendframe       = HandleOnEndFrame;
			m_sCBParams.pfnreleasebitstream = HandleReleaseBitStream;

			m_pCudaEncoder->SetCBFunctions( &m_sCBParams, (void *)m_pCudaEncoder );
		}

		// Now we must create the HW Encoder device
		m_pCudaEncoder->CreateHWEncoder( &m_sEncoderParams );
	}

	CudaFileRender::CudaFileRender(DirectX10Renderer* a_pRenderer, LPCWSTR a_pFilename)
		: CudaTextureRender(a_pRenderer)
	{
	}

	CudaFileRender::~CudaFileRender(void)
	{
		//clean up stuff, release resources etc
		delete m_pCudaEncoder;
		m_pCudaEncoder = NULL;

		// free up resources (device_memory video frame, context lock, CUDA context)
		if (m_sEncoderParams.iUseDeviceMem) {
			cutilDrvSafeCallNoSync( cuvidCtxLock  ( m_cuCtxLock, 0   ) );
			cutilDrvSafeCallNoSync( cuMemFree     ( m_dptrVideoFrame ) );
			cutilDrvSafeCallNoSync( cuvidCtxUnlock( m_cuCtxLock, 0   ) );

			cutilDrvSafeCallNoSync( cuvidCtxLockDestroy(m_cuCtxLock) );
			cutilDrvSafeCallNoSync( cuCtxDestroy(m_cuContext)        );
		}
	}

	void CudaFileRender::present(int effect){
		map((void**)&m_pFrameBuffer, &m_iFrameBufferPitch, 0);
		//m_pCudaEncoder->setSNRData(m_pFrameBuffer);
		if(m_pCudaEncoder)
		{
			NVVE_EncodeFrameParams      efparams;
			efparams.Height           = m_sEncoderParams.iOutputSize[1];
			efparams.Width            = m_sEncoderParams.iOutputSize[0];
			efparams.Pitch            = (m_sEncoderParams.nDeviceMemPitch ? m_sEncoderParams.nDeviceMemPitch : m_sEncoderParams.iOutputSize[0]);
			efparams.PictureStruc     = (NVVE_PicStruct)m_sEncoderParams.iPictureType; 
			efparams.SurfFmt          = (NVVE_SurfaceFormat)m_sEncoderParams.iSurfaceFormat;
			//efparams.SurfFmt          = NV12;
			efparams.progressiveFrame = (m_sEncoderParams.iSurfaceFormat == 3) ? 1 : 0;
			//efparams.progressiveFrame = true;
			efparams.repeatFirstField = 0;
			efparams.topfieldfirst    = (m_sEncoderParams.iSurfaceFormat == 1) ? 1 : 0;

			efparams.picBuf = NULL;
			//efparams.picBuf = (unsigned char *)m_pFrameBuffer;

			if(effect)
				efparams.bLast = false;
			else
				efparams.bLast = true;

			// If m_dptrVideoFrame is NULL, then we assume that frames come from system memory, otherwise it comes from GPU memory
			// VideoEncoder.cpp, EncodeFrame() will automatically copy it to GPU Device memory, if GPU device input is specified
			//if (m_pCudaEncoder->EncodeFrame(efparams, m_dptrVideoFrame, m_cuCtxLock) == false)

			cuda_yuv_transform((void*)m_pFrameBuffer, (void*)m_dptrVideoFrame, efparams.Width, efparams.Height, m_iFrameBufferPitch, m_iPitch);
			if (m_pCudaEncoder->EncodeFrame(efparams, m_dptrVideoFrame) == false)		
			{
				printf("\nEncodeFrame() failed to encode frame\n");
			}
		}
		unmap(0);

		if(!effect) {
			delete m_pCudaEncoder;
			m_pCudaEncoder=NULL;
		}
	}

	void CudaFileRender::setTextureSource(unsigned int a_iTextureNumber, DirectX9Texture* a_pTexture)
	{
		m_pDX9Texture2DSourceRGBA = a_pTexture;
	}

	void CudaFileRender::setTextureSource(unsigned int a_iTextureNumber, DirectX10Texture* a_pTexture)
	{
		m_pDX10Texture2DSourceRGBA = a_pTexture;
	}

	// Parsing the command line arguments and programming the NVEncoderParameters parameters
	bool CudaFileRender::ParseInputParams(NVEncoderParams *pParams, LPCWSTR a_pFilename) 
	{
		int argcount=0;

		pParams->measure_fps  = 0;
		pParams->measure_psnr = 0;
		pParams->force_device = 0;
		pParams->iForcedGPU   = 0;

		// By default we want to do motion estimation on the GPU
		pParams->GPUOffloadLevel= NVVE_GPU_OFFLOAD_ALL; // NVVE_GPU_OFFLOAD_ESTIMATORS;
		pParams->iSurfaceFormat = (int)YV12;
		pParams->iPictureType   = (int)FRAME_PICTURE;

		// This is demo mode, we will print out the help, and run the encode	
		//strcpy(pParams->configFile, VIDEO_CONFIG_FILE);
		//strcpy(pParams->outputFile, VIDEO_OUTPUT_FILE);

		wcscpy(pParams->configFile, VIDEO_CONFIG_FILE);
		wcscpy(pParams->outputFile, a_pFilename);

		// TODO: Guess we are using device memory
		pParams->iUseDeviceMem = 1;

		if(!wcslen(pParams->configFile)) {
			printf("\n *.cfg config file is required to use the encoder\n");
			return false;
		}
		if(!wcslen(pParams->outputFile)) {
			printf("\n *.264 output file is required to use the encoder\n");
			return false;
		}
		return true;
	}

	void CudaFileRender::registerResources(unsigned int nFrames) {
		IDirect3DTexture9* field1=NULL;
		if(this->m_pDX9Texture2DSourceRGBA)
			field1=this->m_pDX9Texture2DSourceRGBA->getTexture();
		IDirect3DTexture9* field2=NULL;

		//TODO: make this foo handle progressive and interlaced frames correct,
		//instead of mapping RGBA and NV12 for testing purposes
		//if(this->m_pDX9Texture2DSourceNV12)
		//	field2=this->m_pDX9Texture2DSourceNV12->getTexture();
		m_ppRegisteredResources[0]=field1;
		//m_ppRegisteredResources[1]=field2;

		CUresult result;
		for (unsigned int i=0; i<nFrames; i++)
			result=cuD3D9RegisterResource(m_ppRegisteredResources[i], 0);
	}

	void CudaFileRender::map(void** ppImageData, unsigned int * pImagePitch, int active_field)
	{
		unsigned int nFrames = m_bIsProgressive ? 1 : 2;

		if(!m_ppRegisteredResources[0])
			registerResources(nFrames);
		CUresult result;
		//cutilDrvSafeCallNoSync ( result = cuD3D9MapResources(nFrames, reinterpret_cast<IDirect3DResource9 **>(m_ppRegisteredResources) ));
		cudaD3D9MapResources(nFrames, (IDirect3DResource9**)m_ppRegisteredResources);

		void* data;
		cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&data, m_ppRegisteredResources[active_field], 0, 0) );

		//cutilDrvSafeCallNoSync ( result = cuD3D9ResourceGetMappedPointer(ppImageData, m_ppRegisteredResources[active_field], 0, 0) );
		//*ppImageData = *((CUdeviceptr*)data)
		*ppImageData = data;
		assert(0 != *ppImageData);

		size_t pPitch;
		size_t pPitchSlice;
		cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPitch(&pPitch, &pPitchSlice, m_ppRegisteredResources[active_field], 0, 0) );
		//cutilDrvSafeCallNoSync ( cuD3D9ResourceGetMappedPitch(pImagePitch, NULL, m_ppRegisteredResources[active_field], 0, 0) );
		*pImagePitch = pPitch;
		assert(0 != *pImagePitch);	
	}

	void CudaFileRender::unmap(int active_field)
	{
		int nFrames = m_bIsProgressive ? 1 : 2;

		//cutilDrvSafeCallNoSync ( cuD3D9UnmapResources(nFrames, reinterpret_cast<IDirect3DResource9 **>(m_ppRegisteredResources) ));
		cudaD3D9UnmapResources(nFrames, (IDirect3DResource9**)m_ppRegisteredResources);
	}
}