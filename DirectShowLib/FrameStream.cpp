//------------------------------------------------------------------------------
// File: FBall.cpp
//
// Desc: DirectShow sample code - implementation of filter behaviors
//       for the bouncing ball source filter.  For more information,
//       refer to Ball.cpp.
//
// Copyright (c) Microsoft Corporation.  All rights reserved.
//------------------------------------------------------------------------------

#include <streams.h>
#include <olectl.h>
#include <initguid.h>

#include "Frame.h"
#include "FrameStream.h"

#pragma warning(disable:4710)  // 'function': function not inlined (optimzation)

namespace TheWhiteAmbit {
	// Setup data

	const AMOVIESETUP_MEDIATYPE sudOpPinTypes =
	{
		&MEDIATYPE_Video,       // Major type
		&MEDIASUBTYPE_NULL      // Minor type
	};

	const AMOVIESETUP_PIN sudOpPin =
	{
		L"Output",              // Pin string name
		FALSE,                  // Is it rendered
		TRUE,                   // Is it an output
		FALSE,                  // Can we have none
		FALSE,                  // Can we have many
		&CLSID_NULL,            // Connects to filter
		NULL,                   // Connects to pin
		1,                      // Number of types
		&sudOpPinTypes 
	};       // Pin details
	const AMOVIESETUP_FILTER sudBallax =
	{
		&CLSID_BouncingBall,    // Filter CLSID
		L"Bouncing Ball",       // String name
		MERIT_DO_NOT_USE,       // Filter merit
		1,                      // Number pins
		&sudOpPin               // Pin details
	};


	// COM global table of objects in this dll

	CFactoryTemplate g_Templates[] = {
		{ L"Bouncing Ball"
		, &CLSID_BouncingBall
		, CFrameSource::CreateInstance
		, NULL
		, &sudBallax }
	};
	int g_cTemplates = sizeof(g_Templates) / sizeof(g_Templates[0]);


	//
	// CreateInstance
	//
	// The only allowed way to create Bouncing balls!
	//
	CUnknown * WINAPI CFrameSource::CreateInstance(LPUNKNOWN lpunk, HRESULT *phr)
	{
		ASSERT(phr);

		CUnknown *punk = new CFrameSource(lpunk, phr);
		if(punk == NULL)
		{
			if(phr)
				*phr = E_OUTOFMEMORY;
		}
		return punk;

	} // CreateInstance


	//
	// Constructor
	//
	// Initialise a CFrameStream object so that we have a pin.
	//
	CFrameSource::CFrameSource(LPUNKNOWN lpunk, HRESULT *phr) :
	CSource(NAME("Bouncing ball"), lpunk, CLSID_BouncingBall)
	{
		ASSERT(phr);
		CAutoLock cAutoLock(&m_cStateLock);

		m_paStreams = (CSourceStream **) new CFrameStream*[1];
		if(m_paStreams == NULL)
		{
			if(phr)
				*phr = E_OUTOFMEMORY;

			return;
		}

		m_paStreams[0] = new CFrameStream(phr, this, L"A Bouncing Ball!");
		if(m_paStreams[0] == NULL)
		{
			if(phr)
				*phr = E_OUTOFMEMORY;

			return;
		}

	} // (Constructor)

	void CFrameSource::setFrameBuffer(Grid<Color>* a_pGrid){
		CFrameStream* stream=(CFrameStream*)m_paStreams[0];
		stream->setFrameBuffer(a_pGrid);
	}


	//
	// Constructor
	//
	CFrameStream::CFrameStream(HRESULT *phr,
		CFrameSource *pParent,
		LPCWSTR pPinName) :
	CSourceStream(NAME("Bouncing Ball"),phr, pParent, pPinName),
		m_iImageWidth(1280),
		m_iImageHeight(1024),
		m_iDefaultRepeatTime(20),
		m_pGrid(NULL)
	{
		ASSERT(phr);
		CAutoLock cAutoLock(&m_cSharedState);

		m_Frame = new CFrame(m_iImageWidth, m_iImageHeight);
		if(m_Frame == NULL)
		{
			if(phr)
				*phr = E_OUTOFMEMORY;
		}

	} // (Constructor)


	//
	// Destructor
	//
	CFrameStream::~CFrameStream()
	{
		CAutoLock cAutoLock(&m_cSharedState);
		if(m_Frame)
			delete m_Frame;

	} // (Destructor)


	//
	// FillBuffer
	//
	// Plots a ball into the supplied video buffer
	//
	HRESULT CFrameStream::FillBuffer(IMediaSample *pms)
	{
		CheckPointer(pms,E_POINTER);
		ASSERT(m_Frame);

		BYTE *pData;
		long lDataLen;

		pms->GetPointer(&pData);
		lDataLen = pms->GetSize();

		ZeroMemory(pData, lDataLen);
		{
			CAutoLock cAutoLockShared(&m_cSharedState);

			// If we haven't just cleared the buffer delete the old
			// ball and move the ball on

			if(m_pGrid)
				m_Frame->PlotBall(pData, m_iPixelSize, m_pGrid);

			// The current time is the sample's start
			CRefTime rtStart = m_rtSampleTime;

			// Increment to find the finish time
			m_rtSampleTime += (LONG)m_iRepeatTime;

			pms->SetTime((REFERENCE_TIME *) &rtStart,(REFERENCE_TIME *) &m_rtSampleTime);
		}

		pms->SetSyncPoint(TRUE);
		return NOERROR;

	} // FillBuffer


	//
	// Notify
	//
	// Alter the repeat rate according to quality management messages sent from
	// the downstream filter (often the renderer).  Wind it up or down according
	// to the flooding level - also skip forward if we are notified of Late-ness
	//
	STDMETHODIMP CFrameStream::Notify(IBaseFilter * pSender, Quality q)
	{
		// Adjust the repeat rate.
		if(q.Proportion<=0)
		{
			m_iRepeatTime = 1000;        // We don't go slower than 1 per second
		}
		else
		{
			m_iRepeatTime = m_iRepeatTime*1000 / q.Proportion;
			if(m_iRepeatTime>1000)
			{
				m_iRepeatTime = 1000;    // We don't go slower than 1 per second
			}
			else if(m_iRepeatTime<10)
			{
				m_iRepeatTime = 10;      // We don't go faster than 100/sec
			}
		}

		// skip forwards
		if(q.Late > 0)
			m_rtSampleTime += q.Late;

		return NOERROR;

	} // Notify


	void CFrameStream::setFrameBuffer(Grid<Color>* a_pGrid){
		if(!m_pGrid)
			m_pGrid = new Grid<Color>(a_pGrid->getWidth(), a_pGrid->getHeight());
		unsigned int width=a_pGrid->getWidth();
		unsigned int height=a_pGrid->getHeight();
		for(unsigned int x=0; x<width; x++)
			for(unsigned int y=0; y<height; y++)
				m_pGrid->setPixel(x, y, a_pGrid->getPixel(x, y));
	}

	//
	// GetMediaType
	//
	// I _prefer_ 5 formats - 8, 16 (*2), 24 or 32 bits per pixel and
	// I will suggest these with an image size of 320x240. However
	// I can accept any image size which gives me some space to bounce.
	//
	// A bit of fun:
	//      8 bit displays get red balls
	//      16 bit displays get blue
	//      24 bit see green
	//      And 32 bit see yellow
	//
	// Prefered types should be ordered by quality, zero as highest quality
	// Therefore iPosition =
	// 0    return a 32bit mediatype
	// 1    return a 24bit mediatype
	// 2    return 16bit RGB565
	// 3    return a 16bit mediatype (rgb555)
	// 4    return 8 bit palettised format
	// (iPosition > 4 is invalid)
	//
	HRESULT CFrameStream::GetMediaType(int iPosition, CMediaType *pmt)
	{
		CheckPointer(pmt,E_POINTER);

		CAutoLock cAutoLock(m_pFilter->pStateLock());
		if(iPosition < 0)
		{
			return E_INVALIDARG;
		}

		// Have we run off the end of types?

		if(iPosition > 4)
		{
			return VFW_S_NO_MORE_ITEMS;
		}

		VIDEOINFO *pvi = (VIDEOINFO *) pmt->AllocFormatBuffer(sizeof(VIDEOINFO));
		if(NULL == pvi)
			return(E_OUTOFMEMORY);

		ZeroMemory(pvi, sizeof(VIDEOINFO));

		switch(iPosition)
		{
		case 0:
			{    
				// Return our highest quality 32bit format

				// since we use RGB888 (the default for 32 bit), there is
				// no reason to use BI_BITFIELDS to specify the RGB
				// masks. Also, not everything supports BI_BITFIELDS

				pvi->bmiHeader.biCompression = BI_RGB;
				pvi->bmiHeader.biBitCount    = 32;
				break;
			}

		case 1:
			{   // Return our 24bit format

				pvi->bmiHeader.biCompression = BI_RGB;
				pvi->bmiHeader.biBitCount    = 24;
				break;
			}
		}

		// (Adjust the parameters common to all formats...)

		// put the optimal palette in place
		for(int i = 0; i < iPALETTE_COLORS; i++)
		{
			pvi->TrueColorInfo.bmiColors[i].rgbRed      = i;
			pvi->TrueColorInfo.bmiColors[i].rgbBlue     = i;
			pvi->TrueColorInfo.bmiColors[i].rgbGreen    = i;
			pvi->TrueColorInfo.bmiColors[i].rgbReserved = 0;
		}

		pvi->bmiHeader.biSize       = sizeof(BITMAPINFOHEADER);
		pvi->bmiHeader.biWidth      = m_iImageWidth;
		pvi->bmiHeader.biHeight     = m_iImageHeight;
		pvi->bmiHeader.biPlanes     = 1;
		pvi->bmiHeader.biSizeImage  = GetBitmapSize(&pvi->bmiHeader);
		pvi->bmiHeader.biClrImportant = 0;

		SetRectEmpty(&(pvi->rcSource)); // we want the whole image area rendered.
		SetRectEmpty(&(pvi->rcTarget)); // no particular destination rectangle

		pmt->SetType(&MEDIATYPE_Video);
		pmt->SetFormatType(&FORMAT_VideoInfo);
		pmt->SetTemporalCompression(FALSE);

		// Work out the GUID for the subtype from the header info.
		const GUID SubTypeGUID = GetBitmapSubtype(&pvi->bmiHeader);
		pmt->SetSubtype(&SubTypeGUID);
		pmt->SetSampleSize(pvi->bmiHeader.biSizeImage);

		return NOERROR;

	} // GetMediaType


	//
	// CheckMediaType
	//
	// We will accept 8, 16, 24 or 32 bit video formats, in any
	// image size that gives room to bounce.
	// Returns E_INVALIDARG if the mediatype is not acceptable
	//
	HRESULT CFrameStream::CheckMediaType(const CMediaType *pMediaType)
	{
		CheckPointer(pMediaType,E_POINTER);

		if((*(pMediaType->Type()) != MEDIATYPE_Video) ||   // we only output video
			!(pMediaType->IsFixedSize()))                   // in fixed size samples
		{                                                  
			return E_INVALIDARG;
		}

		// Check for the subtypes we support
		const GUID *SubType = pMediaType->Subtype();
		if (SubType == NULL)
			return E_INVALIDARG;

		if( //(*SubType != MEDIASUBTYPE_RGB32)   
			(*SubType != MEDIASUBTYPE_RGB24)
			&&(*SubType != MEDIASUBTYPE_NV12)
			//&& (*SubType != MEDIASUBTYPE_YUY2)
			//&& (*SubType != MEDIASUBTYPE_IYUV)
			//&& (*SubType != MEDIASUBTYPE_YV12)
			//&& (*SubType != MEDIASUBTYPE_UYVY)
			)
		{
			return E_INVALIDARG;
		}

		// Get the format area of the media type
		VIDEOINFO *pvi = (VIDEOINFO *) pMediaType->Format();

		if(pvi == NULL)
			return E_INVALIDARG;

		// Check the image size. As my default ball is 10 pixels big
		// look for at least a 20x20 image. This is an arbitary size constraint,
		// but it avoids balls that are bigger than the picture...

		if((pvi->bmiHeader.biWidth < 20) || ( abs(pvi->bmiHeader.biHeight) < 20))
		{
			return E_INVALIDARG;
		}

		// Check if the image width & height have changed
		if(pvi->bmiHeader.biWidth != m_Frame->GetImageWidth() || 
			abs(pvi->bmiHeader.biHeight) != m_Frame->GetImageHeight())
		{
			// If the image width/height is changed, fail CheckMediaType() to force
			// the renderer to resize the image.
			return E_INVALIDARG;
		}


		return S_OK;  // This format is acceptable.

	} // CheckMediaType


	//
	// DecideBufferSize
	//
	// This will always be called after the format has been sucessfully
	// negotiated. So we have a look at m_mt to see what size image we agreed.
	// Then we can ask for buffers of the correct size to contain them.
	//
	HRESULT CFrameStream::DecideBufferSize(IMemAllocator *pAlloc,
		ALLOCATOR_PROPERTIES *pProperties)
	{
		CheckPointer(pAlloc,E_POINTER);
		CheckPointer(pProperties,E_POINTER);

		CAutoLock cAutoLock(m_pFilter->pStateLock());
		HRESULT hr = NOERROR;

		VIDEOINFO *pvi = (VIDEOINFO *) m_mt.Format();
		pProperties->cBuffers = 1;
		pProperties->cbBuffer = pvi->bmiHeader.biSizeImage;

		ASSERT(pProperties->cbBuffer);

		// Ask the allocator to reserve us some sample memory, NOTE the function
		// can succeed (that is return NOERROR) but still not have allocated the
		// memory that we requested, so we must check we got whatever we wanted

		ALLOCATOR_PROPERTIES Actual;
		hr = pAlloc->SetProperties(pProperties,&Actual);
		if(FAILED(hr))
		{
			return hr;
		}

		// Is this allocator unsuitable

		if(Actual.cbBuffer < pProperties->cbBuffer)
		{
			return E_FAIL;
		}

		// Make sure that we have only 1 buffer (we erase the ball in the
		// old buffer to save having to zero a 200k+ buffer every time
		// we draw a frame)

		ASSERT(Actual.cBuffers == 1);
		return NOERROR;

	} // DecideBufferSize


	//
	// SetMediaType
	//
	// Called when a media type is agreed between filters
	//
	HRESULT CFrameStream::SetMediaType(const CMediaType *pMediaType)
	{
		CAutoLock cAutoLock(m_pFilter->pStateLock());

		// Pass the call up to my base class

		HRESULT hr = CSourceStream::SetMediaType(pMediaType);

		if(SUCCEEDED(hr))
		{
			VIDEOINFO * pvi = (VIDEOINFO *) m_mt.Format();
			if (pvi == NULL)
				return E_UNEXPECTED;

			switch(pvi->bmiHeader.biBitCount)
			{
			case 24:    // Make a green pixel
				m_iPixelSize = 3;
				break;

			case 32:    // Make a yellow pixel
				m_iPixelSize = 4;
				break;

			default:
				m_iPixelSize = pvi->bmiHeader.biBitCount / 8;
				break;
			}

			CFrame *pNewBall = new CFrame(pvi->bmiHeader.biWidth, abs(pvi->bmiHeader.biHeight));

			if(pNewBall)
			{
				delete m_Frame;
				m_Frame = pNewBall;
			}
			else
				hr = E_OUTOFMEMORY;

			return NOERROR;
		} 

		return hr;

	} // SetMediaType


	//
	// OnThreadCreate
	//
	// As we go active reset the stream time to zero
	//
	HRESULT CFrameStream::OnThreadCreate()
	{
		CAutoLock cAutoLockShared(&m_cSharedState);
		m_rtSampleTime = 0;

		// we need to also reset the repeat time in case the system
		// clock is turned off after m_iRepeatTime gets very big
		m_iRepeatTime = m_iDefaultRepeatTime;

		return NOERROR;

	} // OnThreadCreate
}

