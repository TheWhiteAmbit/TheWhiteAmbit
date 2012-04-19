
#include "CudaVideoRender.h"

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
	CudaVideoRender::CudaVideoRender(DirectX9Renderer* a_pRenderer, LPCWSTR a_pFilename)
		: CudaTextureRender(a_pRenderer)
	{
		m_bDone       = false;
		m_bAutoQuit   = false;
		m_bUseVsync   = false;
		m_bQAReadback = false;
		m_bFirstFrame = true;
		m_bLoop       = false;
		m_bUpdateCSC  = true;
		m_bUpdateAll  = false;
		m_bInterop    = true;
		m_bReadback   = false; // this flag enables/disables reading back of a video from a window
		m_bIsProgressive = true; // assume it is progressive, unless otherwise noted

		m_pDX9Texture2DField1=NULL;
		m_pDX9Texture2DField2=NULL;

		m_ppRegisteredResources[0]=NULL;
		m_ppRegisteredResources[1]=NULL;

		m_eVideoCreateFlags = cudaVideoCreate_PreferCUVID;
		m_CtxLock = NULL;

		m_cuModNV12toARGB  = 0;
		m_fpNV12toARGB    = 0;
		m_fpPassthru      = 0;

		m_ReadbackSID = 0;
		m_KernelSID = 0;

		m_eColorSpace = ITU601;
		m_nHue        = 0.0f;

		// System Memory surface we want to readback to
		m_bFrameData = new BYTE[2];
		m_bFrameData[0] = 0;
		m_bFrameData[1] = 0;
		m_pFrameQueue   = 0;
		m_pVideoSource  = 0;
		m_pVideoParser  = 0;
		m_pVideoDecoder = 0;

		m_pInteropFrame = new CUdeviceptr[2];
		m_pInteropFrame[0] = 0; // if we're using CUDA malloc
		m_pInteropFrame[1] = 0; // if we're using CUDA malloc


		m_nWindowWidth  = 0;
		m_nWindowHeight = 0;
		m_nVideoWidth  = 0;
		m_nVideoHeight = 0;
		m_nFrameCount = 0;
		m_nDecodeFrameCount = 0;

		setVideoFile(a_pFilename);
	}

	CudaVideoRender::CudaVideoRender(DirectX10Renderer* a_pRenderer, LPCWSTR a_pFilename)
		: CudaTextureRender(a_pRenderer)
	{
	}

	void CudaVideoRender::release(){
		m_pFrameQueue->endDecode();
		m_pVideoSource->stop();

		// clean up CUDA and D3D resources
		cleanup(true);

		//TODO: release textures as long as no good resource management is present :(		
		m_ppRegisteredResources[0]=NULL;
		m_ppRegisteredResources[1]=NULL;

		delete m_pDX9Texture2DField1;
		m_pDX9Texture2DField1=NULL;

		delete m_pDX9Texture2DField2;
		m_pDX9Texture2DField2=NULL;	
	}

	CudaVideoRender::~CudaVideoRender(void)
	{
		//TODO: if not disposed bla bla..
		//dispose();
	}

	void CudaVideoRender::setVideoFile(LPCWSTR a_pVideoFilename)
	{
		m_sFileName=a_pVideoFilename;

		//char* video_file=new char[MAX_PATH];
		//WideCharToMultiByte( CP_ACP, 0, a_pVideoFilename, -1, video_file, MAX_PATH, NULL, NULL );

		// Find out the video size 
		m_bIsProgressive = loadVideoSource(a_pVideoFilename, 
			m_nVideoWidth, m_nVideoHeight, 
			m_nWindowWidth, m_nWindowHeight );
		//delete video_file;

		// Initialize CUDA
		//TODO: dont init cuda twice??
		cuInit(0);

		int bTCC = 0;
		// If we are using TCC driver, then always turn off interop
		if (bTCC) m_bInterop = false;

		// Initialize CUDA/D3D9 context and other video memory resources
		initCudaResources(m_bInterop, bTCC);

		m_pVideoSource->start();
	}

	//void CudaVideoRender::openVideoFile(LPCWSTR a_pVideoFilename){
	//	m_pFrameQueue->endDecode();
	//	m_sFileName=a_pVideoFilename;
	//	reinitCudaResources();
	//}

	bool CudaVideoRender::hasNext(){
		return m_pVideoSource->isStarted() && !m_pFrameQueue->isEndOfDecode();
	}

	void CudaVideoRender::present(int effect){
		if (hasNext()) {
			renderVideoFrame(m_bInterop);
		}
	}

	HRESULT CudaVideoRender::initCudaResources(int bUseInterop, int bTCC)
	{
		HRESULT hr = S_OK;

		CUdevice cuda_device;
		{
			// If we want to use Graphics Interop, then choose the GPU that is capable
			if (bUseInterop) {
				cuda_device = cutilDrvGetMaxGflopsGraphicsDeviceId();
				cutilDrvSafeCallNoSync(cuDeviceGet(&m_cuDevice, cuda_device ));
			} else {
				cuda_device = cutilDrvGetMaxGflopsDeviceId();
				cutilDrvSafeCallNoSync(cuDeviceGet(&m_cuDevice, cuda_device ));
			}
		}

		// get compute capabilities and the devicename
		int major, minor;
		size_t totalGlobalMem;
		char deviceName[256];
		cutilDrvSafeCallNoSync( cuDeviceComputeCapability(&major, &minor, m_cuDevice) );
		cutilDrvSafeCallNoSync( cuDeviceGetName(deviceName, 256, m_cuDevice) );
		printf("> Using GPU Device %d: %s has SM %d.%d compute capability\n", cuda_device, deviceName, major, minor);

		cutilDrvSafeCallNoSync( cuDeviceTotalMem(&totalGlobalMem, m_cuDevice) );
		printf("  Total amount of global memory:     %4.4f MB\n", (float)totalGlobalMem/(1024*1024) );

		// Create CUDA Device w/ D3D9 interop (if WDDM), otherwise CUDA w/o interop (if TCC)
		// (use CU_CTX_BLOCKING_SYNC for better CPU synchronization)
		if (bUseInterop) {
			cutilDrvSafeCallNoSync( cuD3D9CtxCreate(&m_cuContext, &m_cuDevice, CU_CTX_BLOCKING_SYNC, m_pRenderer9->getDevice()) );
		} else {
			cutilDrvSafeCallNoSync( cuCtxCreate(&m_cuContext, CU_CTX_BLOCKING_SYNC, m_cuDevice) );
		}

		// Initialize CUDA releated Driver API (32-bit or 64-bit), depending the platform running
		if (sizeof(void *) == 4) {
			m_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi_Win32.ptx", "./", 2, 2, 2);
		} else {
			m_pCudaModule = new CUmoduleManager("NV12ToARGB_drvapi_x64.ptx", "./", 2, 2, 2);
		}

		m_pCudaModule->GetCudaFunction("NV12ToARGB_drvapi",   &m_fpNV12toARGB);
		m_pCudaModule->GetCudaFunction("Passthru_drvapi",     &m_fpPassthru);

		/////////////////Change///////////////////////////
		// Now we create the CUDA resources and the CUDA decoder context
		initCudaVideo();

		if (bUseInterop) {
			//initD3D9Surface   ( m_pVideoDecoder->targetWidth(), 
			//					m_pVideoDecoder->targetHeight() );
		} else {
			cutilDrvSafeCallNoSync( cuMemAlloc(&m_pInteropFrame[0], m_pVideoDecoder->targetWidth() * m_pVideoDecoder->targetHeight() * 2) );
			cutilDrvSafeCallNoSync( cuMemAlloc(&m_pInteropFrame[1], m_pVideoDecoder->targetWidth() * m_pVideoDecoder->targetHeight() * 2) );
		}

		CUcontext cuCurrent = NULL;
		CUresult result = cuCtxPopCurrent(&cuCurrent);
		if (result != CUDA_SUCCESS) {
			printf("cuCtxPopCurrent: %d\n", result);
			assert(0);
		}

		/////////////////////////////////////////
		return ((m_pCudaModule && m_pVideoDecoder) ? S_OK : E_FAIL);
	}

	HRESULT CudaVideoRender::reinitCudaResources()
	{
		// Free resources
		cleanup(false);

		// Reinit VideoSource and Frame Queue
		m_bIsProgressive = loadVideoSource(m_sFileName, m_nVideoWidth, m_nVideoHeight, m_nWindowWidth, m_nWindowHeight );
		//m_bIsProgressive = loadVideoSource(m_sFileName.c_str(), m_nVideoWidth, m_nVideoHeight, m_nWindowWidth, m_nWindowHeight );

		/////////////////Change///////////////////////////
		initCudaVideo     ( );
		/////////////////////////////////////////

		return S_OK;
	}

	bool CudaVideoRender::loadVideoSource(const wchar_t *video_file, 
		unsigned int &width    , unsigned int &height,
		unsigned int &dispWidth, unsigned int &dispHeight)
	{
		std::auto_ptr<FrameQueue> apFrameQueue(new FrameQueue);
		std::auto_ptr<VideoSource> apVideoSource(new VideoSource(video_file, apFrameQueue.get()));

		// retrieve the video source (width,height)    
		apVideoSource->getSourceDimensions(width, height);
		apVideoSource->getSourceDimensions(dispWidth, dispHeight);

		std::cout << apVideoSource->format() << std::endl;

		m_pFrameQueue  = apFrameQueue.release();
		m_pVideoSource = apVideoSource.release();

		if (m_pVideoSource->format().codec == cudaVideoCodec_JPEG ||
			m_pVideoSource->format().codec == cudaVideoCodec_MPEG2) 
		{
			m_eVideoCreateFlags = cudaVideoCreate_PreferCUDA;
		}

		bool IsProgressive = 0;
		m_pVideoSource->getProgressive(IsProgressive);
		return IsProgressive;
	}

	void CudaVideoRender::initCudaVideo( )
	{
		// bind the context lock to the CUDA context
		CUresult result = cuvidCtxLockCreate(&m_CtxLock, m_cuContext);
		if (result != CUDA_SUCCESS) {
			printf("cuvidCtxLockCreate failed: %d\n", result);
			assert(0);
		}

		std::auto_ptr<VideoDecoder> apVideoDecoder(new VideoDecoder(m_pVideoSource->format(), m_cuContext, m_eVideoCreateFlags, m_CtxLock));
		std::auto_ptr<VideoParser> apVideoParser(new VideoParser(apVideoDecoder.get(), m_pFrameQueue));
		m_pVideoSource->setParser(*apVideoParser.get());

		m_pVideoParser  = apVideoParser.release();
		m_pVideoDecoder = apVideoDecoder.release();

		// Create a Stream ID for handling Readback
		if (m_bReadback) {
			cutilDrvSafeCallNoSync( cuStreamCreate(&m_ReadbackSID, 0) );
			cutilDrvSafeCallNoSync( cuStreamCreate(&m_KernelSID,   0) );
			printf("> initCudaVideo()\n");
			printf("  CUDA Streams (%s) <m_ReadbackSID = %p>\n", ((m_ReadbackSID == 0) ? "Disabled" : "Enabled"), m_ReadbackSID );
			printf("  CUDA Streams (%s) <m_KernelSID   = %p>\n", ((m_KernelSID   == 0) ? "Disabled" : "Enabled"), m_KernelSID   );
		}
	}


	void CudaVideoRender::terminateCudaVideo(bool bDestroyContext)
	{
		if (m_pVideoParser)  delete m_pVideoParser;
		if (m_pVideoDecoder) delete m_pVideoDecoder;
		if (m_pVideoSource)  delete m_pVideoSource;
		if (m_pFrameQueue)   delete m_pFrameQueue;

		if (m_CtxLock) {
			cutilDrvSafeCallNoSync( cuvidCtxLockDestroy(m_CtxLock) );
		}
		if (m_cuContext && bDestroyContext)  {
			cutilDrvSafeCallNoSync( cuCtxDestroy(m_cuContext) );
			m_cuContext = NULL;
		}

		if (m_ReadbackSID)   cuStreamDestroy(m_ReadbackSID);
		if (m_KernelSID)     cuStreamDestroy(m_KernelSID);
	}

	// Run the Cuda part of the computation
	bool CudaVideoRender::copyDecodedFrameToTexture(unsigned int &nRepeats, int bUseInterop, int *pbIsProgressive)
	{
		CUVIDPARSERDISPINFO oDisplayInfo;
		if (m_pFrameQueue->dequeue(&oDisplayInfo))
		{
			CCtxAutoLock lck  ( m_CtxLock );
			// Push the current CUDA context (only if we are using CUDA decoding path)
			CUresult result = cuCtxPushCurrent(m_cuContext);

			CUdeviceptr	 pDecodedFrame[2] = { 0, 0 };
			CUdeviceptr  pInteropFrame[2] = { 0, 0 };

			int num_fields = (oDisplayInfo.progressive_frame ? (1) : (2+oDisplayInfo.repeat_first_field));
			*pbIsProgressive = oDisplayInfo.progressive_frame;
			m_bIsProgressive = oDisplayInfo.progressive_frame ? true : false;
			for (int active_field=0; active_field<num_fields; active_field++)
			{
				nRepeats = oDisplayInfo.repeat_first_field;
				CUVIDPROCPARAMS oVideoProcessingParameters;
				memset(&oVideoProcessingParameters, 0, sizeof(CUVIDPROCPARAMS));

				oVideoProcessingParameters.progressive_frame = oDisplayInfo.progressive_frame;
				oVideoProcessingParameters.second_field      = active_field;
				oVideoProcessingParameters.top_field_first   = oDisplayInfo.top_field_first;
				oVideoProcessingParameters.unpaired_field    = (num_fields == 1);

				unsigned int nDecodedPitch = 0;
				unsigned int nWidth = 0;
				unsigned int nHeight = 0;

				// map decoded video frame to CUDA surfae
				m_pVideoDecoder->mapFrame(oDisplayInfo.picture_index, (unsigned int*)&pDecodedFrame[active_field], &nDecodedPitch, &oVideoProcessingParameters);
				nWidth  = m_pVideoDecoder->targetWidth();
				nHeight = m_pVideoDecoder->targetHeight();
				// map DirectX texture to CUDA surface
				unsigned int nTexturePitch = 0;

				// If we are Encoding and this is the 1st Frame, we make sure we allocate system memory for readbacks
				if (m_bReadback && m_bFirstFrame && m_ReadbackSID) {
					CUresult result;
					cutilDrvSafeCallNoSync( result = cuMemAllocHost( (void **)&m_bFrameData[0], (nDecodedPitch * nHeight * 3 / 2) ) );
					cutilDrvSafeCallNoSync( result = cuMemAllocHost( (void **)&m_bFrameData[1], (nDecodedPitch * nHeight * 3 / 2) ) );
					m_bFirstFrame = false;
					if (result != CUDA_SUCCESS) {
						printf("cuMemAllocHost returned %d\n", (int)result);
					}
				}

				// If streams are enabled, we can perform the readback to the host while the kernel is executing
				if (m_bReadback && m_ReadbackSID) {
					//TODO: test if &m_bFrameData[active_field] is the correct void*
					CUresult result = cuMemcpyDtoHAsync(&m_bFrameData[active_field], pDecodedFrame[active_field], (nDecodedPitch * nHeight * 3 / 2), m_ReadbackSID);
					if (result != CUDA_SUCCESS) {
						printf("cuMemAllocHost returned %d\n", (int)result);
					}
				}

#if ENABLE_DEBUG_OUT
				printf("%s = %02d, PicIndex = %02d, OutputPTS = %08d\n", 
					(oDisplayInfo.progressive_frame ? "Frame" : "Field"),
					m_nDecodeFrameCount, oDisplayInfo.picture_index, oDisplayInfo.timestamp);
#endif

				if (true) {
					// map the texture surface
					//m_pImageDX->map(&pInteropFrame[active_field], &nTexturePitch, active_field);
					//TODO: map interop frames to d3d9surface
					map(&pInteropFrame[active_field], &nTexturePitch, active_field);
				} else {
					pInteropFrame[active_field] = m_pInteropFrame[active_field];
					nTexturePitch = m_pVideoDecoder->targetWidth() * 2; 
				}

				// perform post processing on the CUDA surface (performs colors space conversion and post processing)
				// comment this out if we inclue the line of code seen above 
				cudaPostProcessFrame(&pDecodedFrame[active_field], nDecodedPitch, &pInteropFrame[active_field], nTexturePitch, m_pCudaModule->getModule(), m_fpNV12toARGB, m_KernelSID);
				if (true) {
					// unmap the texture surface
					//m_pImageDX->unmap(active_field);
					//TODO: map interop frames to d3d9surface
					unmap(active_field);
				}

				// unmap video frame
				// unmapFrame() synchronizes with the VideoDecode API (ensures the frame has finished decoding)
				m_pVideoDecoder->unmapFrame((unsigned int*)&pDecodedFrame[active_field]);
				// release the frame, so it can be re-used in decoder
				m_pFrameQueue->releaseFrame(&oDisplayInfo);
				m_nDecodeFrameCount++;
			}

			// Detach from the Current thread
			cutilDrvSafeCallNoSync( cuCtxPopCurrent(NULL) );
		} else {
			return false;
		}

		// check if decoding has come to an end.
		// if yes, signal the app to shut down.
		if (!m_pVideoSource->isStarted() || m_pFrameQueue->isEndOfDecode())
		{
			// Let's free the Frame Data
			if (m_ReadbackSID && m_bFrameData) {
				cuMemFreeHost((void *)m_bFrameData[0]);
				cuMemFreeHost((void *)m_bFrameData[1]);
				m_bFrameData[0] = NULL;
				m_bFrameData[1] = NULL;
			}

			// Let's just stop, and allow the user to quit, so they can at least see the results
			m_pVideoSource->stop();

			// If we want to loop reload the video file and restart
			if (m_bLoop && !m_bAutoQuit) {
				reinitCudaResources();
				m_nFrameCount = 0;
				m_nDecodeFrameCount = 0;
				m_pVideoSource->start();
			}
			if (m_bAutoQuit) {
				m_bDone = true;
			}
		}
		return true;
	}

	// This is the CUDA stage for Video Post Processing.  Last stage takes care of the NV12 to ARGB
	void CudaVideoRender::cudaPostProcessFrame(CUdeviceptr * ppDecodedFrame, unsigned int nDecodedPitch, 
		CUdeviceptr * ppTextureData,  unsigned int nTexturePitch,
		CUmodule m_cuModNV12toARGB, 
		CUfunction fpCudaKernel, CUstream streamID)
	{
		uint32 nWidth  = m_pVideoDecoder->targetWidth();
		uint32 nHeight = m_pVideoDecoder->targetHeight();

		// Upload the Color Space Conversion Matrices
		if (m_bUpdateCSC) {
			// CCIR 601/709
			float hueColorSpaceMat[9];
			setColorSpaceMatrix (m_eColorSpace,    hueColorSpaceMat, m_nHue);
			updateConstantMemory_drvapi( m_cuModNV12toARGB, hueColorSpaceMat );
			if (!m_bUpdateAll) 
				m_bUpdateCSC = false;
		}

		// TODO: Stage for handling video post processing

		// Final Stage: NV12toARGB color space conversion
		CUresult eResult;
		eResult = cudaLaunchNV12toARGBDrv  (*ppDecodedFrame, nDecodedPitch,
			*ppTextureData,  nTexturePitch,
			nWidth, nHeight, fpCudaKernel, streamID);
	}

	// Release all previously initd objects
	HRESULT CudaVideoRender::cleanup(bool bDestroyContext)
	{
		// Attach the CUDA Context (so we may properly free memroy)
		cutilDrvSafeCallNoSync( cuCtxPushCurrent(m_cuContext) );

		if (m_pInteropFrame[0]) {
			cutilDrvSafeCallNoSync( cuMemFree(m_pInteropFrame[0]) );
		}
		if (m_pInteropFrame[1]) {
			cutilDrvSafeCallNoSync( cuMemFree(m_pInteropFrame[1]) );
		}
		// Detach from the Current thread
		cutilDrvSafeCallNoSync( cuCtxPopCurrent(NULL) );
		terminateCudaVideo(bDestroyContext);
		return S_OK;
	}

	// Launches the CUDA kernels to fill in the texture data
	void CudaVideoRender::renderVideoFrame(bool bUseInterop )
	{
		static unsigned int nRepeatFrame = 0;
		int bIsProgressive = 1, bFPSComputed = 0;
		bool bFramesDecoded = false;

		if (nRepeatFrame > 0)
		{
			--nRepeatFrame;
		}
		if (0 != m_pFrameQueue)
		{
			bFramesDecoded = copyDecodedFrameToTexture(nRepeatFrame, true, &bIsProgressive);		
		}
	}

	unsigned int CudaVideoRender::getVideoWidth(){
		return m_nVideoWidth;
	}
	unsigned int CudaVideoRender::getVideoHeight(){
		return m_nVideoHeight;
	}

	void CudaVideoRender::setTextureTarget(unsigned int a_iTextureNumber, DirectX9Texture* a_pTexture)
	{
		switch(a_iTextureNumber) {
		case 0:
			{
				this->m_pDX9Texture2DField1=a_pTexture;
				IDirect3DTexture9* tex=this->m_pDX9Texture2DField1->getTexture();			
			}
			break;
		case 1:
			{
				this->m_pDX9Texture2DField2=a_pTexture;
				IDirect3DTexture9* tex=this->m_pDX9Texture2DField2->getTexture();
			}
			break;		
		default:
			break;
		}
	}

	DirectX9Texture* CudaVideoRender::getTextureTarget(unsigned int a_iTextureNumber)
	{
		switch(a_iTextureNumber) {
		case 0:
			{
				return this->m_pDX9Texture2DField1;		
			}
			break;
		case 1:
			{
				return this->m_pDX9Texture2DField2;
			}
			break;		
		default:
			return NULL;
			break;
		}
	}

	void CudaVideoRender::registerResources(unsigned int nFrames) {
		IDirect3DTexture9* field1=NULL;
		if(this->m_pDX9Texture2DField1)
			field1=this->m_pDX9Texture2DField1->getTexture();
		IDirect3DTexture9* field2=NULL;
		if(this->m_pDX9Texture2DField2)
			field2=this->m_pDX9Texture2DField2->getTexture();
		m_ppRegisteredResources[0]=field1;
		m_ppRegisteredResources[1]=field2;

		for (unsigned int i=0; i<nFrames; i++)
			CUresult resultReg=cuD3D9RegisterResource(m_ppRegisteredResources[i], 0);
	}

	void CudaVideoRender::map(CUdeviceptr * ppImageData, unsigned int * pImagePitch, int active_field)
	{
		unsigned int nFrames = m_bIsProgressive ? 1 : 2;

		if(!m_ppRegisteredResources[0])
			registerResources(nFrames);

		cutilDrvSafeCallNoSync ( cuD3D9MapResources(nFrames, reinterpret_cast<IDirect3DResource9 **>(m_ppRegisteredResources) ));

		cutilDrvSafeCallNoSync ( cuD3D9ResourceGetMappedPointer(ppImageData, m_ppRegisteredResources[active_field], 0, 0) );
		assert(0 != *ppImageData);

		cutilDrvSafeCallNoSync ( cuD3D9ResourceGetMappedPitch(pImagePitch, NULL, m_ppRegisteredResources[active_field], 0, 0) );
		assert(0 != *pImagePitch);	
	}

	void CudaVideoRender::unmap(int active_field)
	{
		int nFrames = m_bIsProgressive ? 1 : 2;

		cutilDrvSafeCallNoSync ( cuD3D9UnmapResources(nFrames, reinterpret_cast<IDirect3DResource9 **>(m_ppRegisteredResources) ));
	}
}