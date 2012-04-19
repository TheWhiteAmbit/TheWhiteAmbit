#pragma once

#include <Windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <Mfreadwrite.h>
#include <mferror.h>

#pragma comment(lib, "mfreadwrite")
#pragma comment(lib, "mfplat")
#pragma comment(lib, "mfuuid")

#include "../SceneGraphLib/Grid.h"
#include "../SceneGraphLib/Color.h"
#include "../DirectX9Lib/DirectX9Renderer.h"

namespace TheWhiteAmbit {
	class MediaFoundationFileRender
	{
		HRESULT hr;
		// Reference to the Sinkwriter
		IMFSinkWriter *pSinkWriter;
		// Pointer to the Stream
		DWORD stream;

		// Format constants
		UINT32 VIDEO_WIDTH;
		UINT32 VIDEO_HEIGHT;
		UINT32 VIDEO_FPS;
		UINT32 VIDEO_BIT_RATE;
		GUID   VIDEO_ENCODING_FORMAT;
		GUID   VIDEO_INPUT_FORMAT;
		UINT32 VIDEO_PELS;

		// Buffer to hold the video frame data.
		DWORD* videoFrameBuffer;

		void setAllPixelsGreen();
		HRESULT InitializeSinkWriter(IMFSinkWriter **ppWriter, DWORD *pStreamIndex, LPCWSTR a_pFilename);
		HRESULT WriteFrame(
			IMFSinkWriter *pWriter, 
			DWORD streamIndex, 
			const LONGLONG& rtStart,        // Time stamp.
			const LONGLONG& rtDuration      // Frame duration.
			);

		LONGLONG rtStart;
	public:
		MediaFoundationFileRender(LPCWSTR a_pFilename, unsigned int width, unsigned int height);
		virtual ~MediaFoundationFileRender(void);

		virtual void present(int effect);
		virtual void setFrameBuffer(Grid<Color>* a_pGrid);
	};
}