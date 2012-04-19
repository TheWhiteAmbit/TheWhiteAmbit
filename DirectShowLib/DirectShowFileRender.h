#pragma once


#ifdef _DEBUG
#pragma comment(lib, "strmbasd")
#pragma comment(lib, "msvcrtd")
//#pragma comment(lib, "atlsd")
#else
#pragma comment(lib, "strmbase")
#pragma comment(lib, "msvcrt")
//#pragma comment(lib, "atls")
#endif

#pragma comment(lib, "winmm")
#pragma comment(lib, "strmiids")
#pragma comment(lib, "wmcodecdspuuid")


template <class T> void SafeRelease(T **ppT)
{
	if (*ppT)
	{
		(*ppT)->Release();
		*ppT = NULL;
	}
}

//#include <windows.h>
//#include <dshow.h>

#include <streams.h>

//#include <amstream.h>
//#include <dvdmedia.h>
//#include <mmsystem.h>
//#include <atlbase.h>
//#include <stdio.h>
//#include <mtype.h>
//#include <wxdebug.h>
//#include <reftime.h>
//#include <strmif.h>
//#include <mtype.h>
//#include <wxdebug.h>
//#include <wxlist.h>
//#include <wxutil.h>
//#include <reftime.h>
//#include <combase.h>
//#include <amfilter.h>
//#include <source.h>


#include "../CudaLib/rendering.h"
#include "../SceneGraphLib/Grid.h"
#include "../DirectX9Lib/DirectX9Renderer.h"

#include "Frame.h"
#include "FrameStream.h"
namespace TheWhiteAmbit {
	class DirectShowFileRender
	{
		HRESULT hr;
		HANDLE hStdout;
		DWORD dwRegister;

		LONGLONG total_source_frames;

		IFilterGraph *pFilterGraph;
		IGraphBuilder *pGraphBuilder;
		//CComPtr<IGraphBuilder> pGraphBuilder;
		IMediaControl *pMediaControl;
		IMediaSeeking *pSeek;
		IMediaEvent *pMediaEvent;
		IMediaFilter *pMediaFilter;
		IBaseFilter *pEncoder;
		IBaseFilter *pFileWriter;
		IBaseFilter *pColorConverter;
		IFileSinkFilter *pSink;

#ifdef USE_MUXER
		IBaseFilter *pMuxingFilter;
#endif	
#ifdef USE_FILESOURCE
		IBaseFilter *pFileSource;
#endif
#ifndef USE_FILESOURCE
		CFrameSource *pFrameSource;
#endif

	public:
		DirectShowFileRender(LPCWSTR a_pFilename);
		virtual ~DirectShowFileRender(void);

		virtual void present(int effect);
		virtual void setFrameBuffer(Grid<Color>* a_pGrid);
	};
}