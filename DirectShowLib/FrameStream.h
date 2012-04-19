//------------------------------------------------------------------------------
// File: FBall.h
//
// Desc: DirectShow sample code - main header file for the bouncing ball
//       source filter.  For more information refer to Ball.cpp
//
// Copyright (c) Microsoft Corporation.  All rights reserved.
//------------------------------------------------------------------------------

#pragma once

#include "../SceneGraphLib/Grid.h"
#include "../DirectX9Lib/DirectX9Renderer.h"


namespace TheWhiteAmbit {
	//------------------------------------------------------------------------------
	// Forward Declarations
	//------------------------------------------------------------------------------
	// The class managing the output pin
	class CFrameStream;


	//------------------------------------------------------------------------------
	// Class CFrameSource
	//
	// This is the main class for the bouncing ball filter. It inherits from
	// CSource, the DirectShow base class for source filters.
	//------------------------------------------------------------------------------
	class CFrameSource : public CSource
	{
	public:

		// The only allowed way to create Bouncing balls!
		static CUnknown * WINAPI CreateInstance(LPUNKNOWN lpunk, HRESULT *phr);
		void setFrameBuffer(Grid<Color>* a_pGrid);
	private:

		// It is only allowed to to create these objects with CreateInstance
		CFrameSource(LPUNKNOWN lpunk, HRESULT *phr);

	}; // CFrameSource


	//------------------------------------------------------------------------------
	// Class CFrameStream
	//
	// This class implements the stream which is used to output the bouncing ball
	// data from the source filter. It inherits from DirectShows's base
	// CSourceStream class.
	//------------------------------------------------------------------------------
	class CFrameStream : public CSourceStream
	{

	public:

		CFrameStream(HRESULT *phr, CFrameSource *pParent, LPCWSTR pPinName);
		~CFrameStream();

		// plots a ball into the supplied video frame
		HRESULT FillBuffer(IMediaSample *pms);

		// Ask for buffers of the size appropriate to the agreed media type
		HRESULT DecideBufferSize(IMemAllocator *pIMemAlloc,
			ALLOCATOR_PROPERTIES *pProperties);

		// Set the agreed media type, and set up the necessary ball parameters
		HRESULT SetMediaType(const CMediaType *pMediaType);

		// Because we calculate the ball there is no reason why we
		// can't calculate it in any one of a set of formats...
		HRESULT CheckMediaType(const CMediaType *pMediaType);
		HRESULT GetMediaType(int iPosition, CMediaType *pmt);

		// Resets the stream time to zero
		HRESULT OnThreadCreate(void);

		// Quality control notifications sent to us
		STDMETHODIMP Notify(IBaseFilter * pSender, Quality q);

		void setFrameBuffer(Grid<Color>* a_pGrid);
	private:

		int m_iImageHeight;                 // The current image height
		int m_iImageWidth;                  // And current image width
		int m_iRepeatTime;                  // Time in msec between frames
		const int m_iDefaultRepeatTime;     // Initial m_iRepeatTime

		//BYTE m_BallPixel[4];                // Represents one coloured ball
		int m_iPixelSize;                   // The pixel size in bytes
		//PALETTEENTRY m_Palette[256];        // The optimal palette for the image
		Grid<Color>* m_pGrid;

		CCritSec m_cSharedState;            // Lock on m_rtSampleTime and m_Ball
		CRefTime m_rtSampleTime;            // The time stamp for each sample
		CFrame *m_Frame;                      // The current ball object

		// set up the palette appropriately
		//enum Colour {Red, Blue, Green, Yellow};
		//HRESULT SetPaletteEntries(Colour colour);

	}; // CFrameStream

}