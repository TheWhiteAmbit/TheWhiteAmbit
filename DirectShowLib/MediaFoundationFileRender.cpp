
#include "MediaFoundationFileRender.h"

namespace TheWhiteAmbit {

	template <class T> void SafeRelease(T **ppT)
	{
		if (*ppT)
		{
			(*ppT)->Release();
			*ppT = NULL;
		}
	}


	MediaFoundationFileRender::MediaFoundationFileRender(LPCWSTR a_pFilename, unsigned int width, unsigned int height)
	{	
		VIDEO_WIDTH = width;
		VIDEO_HEIGHT = height;
		VIDEO_FPS = 60;
		//VIDEO_BIT_RATE = 800000;
		//VIDEO_BIT_RATE = 20000000;
		VIDEO_BIT_RATE = 320000000;

		VIDEO_ENCODING_FORMAT = MFVideoFormat_WVC1;
		//MFVideoFormat_MPEG2;
		//MFVideoFormat_WMV3;
		//MFVideoFormat_H264
		//MFVideoFormat_WVC1
		VIDEO_INPUT_FORMAT = MFVideoFormat_RGB32;
		//MFVideoFormat_RGB32;
		//MFVideoFormat_NV12;
		VIDEO_PELS = VIDEO_WIDTH * VIDEO_HEIGHT;

		videoFrameBuffer = new DWORD[VIDEO_PELS];

		setAllPixelsGreen();
		rtStart = 0;

		hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
		if (SUCCEEDED(hr))
		{
			hr = MFStartup(MF_VERSION);
			if (SUCCEEDED(hr))
			{
				pSinkWriter = NULL;            
				hr = InitializeSinkWriter(&pSinkWriter, &stream, a_pFilename);            
			}
		}

	}

	MediaFoundationFileRender::~MediaFoundationFileRender(void)
	{
		if (SUCCEEDED(hr))
		{
			hr = pSinkWriter->Finalize();		
			SafeRelease(&pSinkWriter);
			MFShutdown();
		}
		CoUninitialize();
	}

	void MediaFoundationFileRender::present(int effect) {
		if (SUCCEEDED(hr))
		{
			// Send frames to the sink writer.		
			UINT64 rtDuration;

			MFFrameRateToAverageTimePerFrame(VIDEO_FPS, 1, &rtDuration);

			for (int i = 0; i < effect; i++)
			{
				hr = WriteFrame(pSinkWriter, stream, rtStart, rtDuration);
				if (FAILED(hr))
				{
					break;
				}
				rtStart += rtDuration;
			}
		}
	}

	void MediaFoundationFileRender::setFrameBuffer(Grid<Color>* a_pGrid){
		BYTE* pFrameBuffer=(BYTE*)videoFrameBuffer;
		unsigned int width=a_pGrid->getWidth();
		unsigned int height=a_pGrid->getHeight();
		for ( unsigned int x = 0; x < width; x++ ) {
			for ( unsigned int y = 0; y < height; y++ ) {
				//TODO: inspect all the twists in texture memory layout
				//D3DXVECTOR4 vecColor=a_pGrid->getPixel( width - x - 1, height - y - 1);
				D3DXVECTOR4 vecColor=a_pGrid->getPixel( width - x - 1, y );
				BYTE color_r = max(0, min(255, (int)(vecColor.x*255.0f*vecColor.w)));
				BYTE color_g = max(0, min(255, (int)(vecColor.y*255.0f*vecColor.w)));
				BYTE color_b = max(0, min(255, (int)(vecColor.z*255.0f*vecColor.w)));
				BYTE color_a = max(0, min(255, (int)(vecColor.w*255.0f)));
				unsigned long base=(x+y*width)*4;
				pFrameBuffer[base]=color_b;
				pFrameBuffer[base+1]=color_g;
				pFrameBuffer[base+2]=color_r;
				pFrameBuffer[base+3]=color_a;
			}
		}			
	}

	void MediaFoundationFileRender::setAllPixelsGreen(){
		// Set all pixels to green
		for (DWORD i = 0; i < VIDEO_PELS; ++i)
		{
			videoFrameBuffer[i] = 0x0000FF00;
		}
	}

	HRESULT MediaFoundationFileRender::InitializeSinkWriter(IMFSinkWriter **ppWriter, DWORD *pStreamIndex, LPCWSTR a_pFilename)
	{
		*ppWriter = NULL;
		*pStreamIndex = NULL;

		IMFSinkWriter   *pSinkWriter = NULL;
		IMFMediaType    *pMediaTypeOut = NULL;   
		IMFMediaType    *pMediaTypeIn = NULL;   
		DWORD           streamIndex;

		HRESULT hr = MFCreateSinkWriterFromURL(a_pFilename, NULL, NULL, &pSinkWriter);
		//MFCreateMPEG4MediaSink(
		//MFCreateSinkWriterFromMediaSink(

		// Set the output media type.
		if (SUCCEEDED(hr))
		{
			hr = MFCreateMediaType(&pMediaTypeOut);   
		}
		if (SUCCEEDED(hr))
		{
			hr = pMediaTypeOut->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);     
		}
		if (SUCCEEDED(hr))
		{
			hr = pMediaTypeOut->SetGUID(MF_MT_SUBTYPE, VIDEO_ENCODING_FORMAT);   
		}
		if (SUCCEEDED(hr))
		{
			hr = pMediaTypeOut->SetUINT32(MF_MT_AVG_BITRATE, VIDEO_BIT_RATE);   
		}
		if (SUCCEEDED(hr))
		{
			hr = pMediaTypeOut->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);   
		}
		if (SUCCEEDED(hr))
		{
			hr = MFSetAttributeSize(pMediaTypeOut, MF_MT_FRAME_SIZE, VIDEO_WIDTH, VIDEO_HEIGHT);   
		}
		if (SUCCEEDED(hr))
		{
			hr = MFSetAttributeRatio(pMediaTypeOut, MF_MT_FRAME_RATE, VIDEO_FPS, 1);   
		}
		if (SUCCEEDED(hr))
		{
			hr = MFSetAttributeRatio(pMediaTypeOut, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);   
		}
		if (SUCCEEDED(hr))
		{
			hr = pSinkWriter->AddStream(pMediaTypeOut, &streamIndex);   
		}

		// Set the input media type.
		if (SUCCEEDED(hr))
		{
			hr = MFCreateMediaType(&pMediaTypeIn);   
		}
		if (SUCCEEDED(hr))
		{
			hr = pMediaTypeIn->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);   
		}
		if (SUCCEEDED(hr))
		{
			hr = pMediaTypeIn->SetGUID(MF_MT_SUBTYPE, VIDEO_INPUT_FORMAT);     
		}
		if (SUCCEEDED(hr))
		{
			//hr = pMediaTypeIn->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_MixedInterlaceOrProgressive);   		
			hr = pMediaTypeIn->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
		}
		if (SUCCEEDED(hr))
		{
			hr = MFSetAttributeSize(pMediaTypeIn, MF_MT_FRAME_SIZE, VIDEO_WIDTH, VIDEO_HEIGHT);
		}
		if (SUCCEEDED(hr))
		{
			hr = MFSetAttributeRatio(pMediaTypeIn, MF_MT_FRAME_RATE, VIDEO_FPS, 1);
		}
		if (SUCCEEDED(hr))
		{
			hr = MFSetAttributeRatio(pMediaTypeIn, MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
		}
		if (SUCCEEDED(hr))
		{
			hr = pSinkWriter->SetInputMediaType(streamIndex, pMediaTypeIn, NULL);
		}

		// Tell the sink writer to start accepting data.
		if (SUCCEEDED(hr))
		{
			hr = pSinkWriter->BeginWriting();
		}

		// Return the pointer to the caller.
		if (SUCCEEDED(hr))
		{
			*ppWriter = pSinkWriter;
			(*ppWriter)->AddRef();
			*pStreamIndex = streamIndex;
		}

		SafeRelease(&pSinkWriter);
		SafeRelease(&pMediaTypeOut);
		SafeRelease(&pMediaTypeIn);
		return hr;
	}


	HRESULT MediaFoundationFileRender::WriteFrame(
		IMFSinkWriter *pWriter, 
		DWORD streamIndex, 
		const LONGLONG& rtStart,        // Time stamp.
		const LONGLONG& rtDuration      // Frame duration.
		)
	{
		IMFSample *pSample = NULL;
		IMFMediaBuffer *pBuffer = NULL;

		const LONG cbWidth = 4 * VIDEO_WIDTH;
		const DWORD cbBuffer = cbWidth * VIDEO_HEIGHT;

		BYTE *pData = NULL;

		// Create a new memory buffer.
		HRESULT hr = MFCreateMemoryBuffer(cbBuffer, &pBuffer);

		// Lock the buffer and copy the video frame to the buffer.
		if (SUCCEEDED(hr))
		{
			hr = pBuffer->Lock(&pData, NULL, NULL);
		}
		if (SUCCEEDED(hr))
		{
			hr = MFCopyImage(
				pData,                      // Destination buffer.
				cbWidth,                    // Destination stride.
				(BYTE*)videoFrameBuffer,    // First row in source image.
				cbWidth,                    // Source stride.
				cbWidth,                    // Image width in bytes.
				VIDEO_HEIGHT                // Image height in pixels.
				);
		}
		if (pBuffer)
		{
			pBuffer->Unlock();
		}

		// Set the data length of the buffer.
		if (SUCCEEDED(hr))
		{
			hr = pBuffer->SetCurrentLength(cbBuffer);
		}

		// Create a media sample and add the buffer to the sample.
		if (SUCCEEDED(hr))
		{
			hr = MFCreateSample(&pSample);
		}
		if (SUCCEEDED(hr))
		{
			hr = pSample->AddBuffer(pBuffer);
		}

		// Set the time stamp and the duration.
		if (SUCCEEDED(hr))
		{
			hr = pSample->SetSampleTime(rtStart);
		}
		if (SUCCEEDED(hr))
		{
			hr = pSample->SetSampleDuration(rtDuration);
		}

		// Send the sample to the Sink Writer.
		if (SUCCEEDED(hr))
		{
			hr = pWriter->WriteSample(streamIndex, pSample);
		}

		SafeRelease(&pSample);
		SafeRelease(&pBuffer);
		return hr;
	}
}