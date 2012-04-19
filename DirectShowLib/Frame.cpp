//------------------------------------------------------------------------------
// File: Ball.cpp
//
// Desc: DirectShow sample code.  This sample illustrates a simple source 
//       filter that produces decompressed images showing a ball bouncing 
//       around. Each movement of the ball is done by generating a new image. 
//       We use the CSource and CSourceStream base classes to manage a source 
//       filter - we are a live source and so do not support any seeking.
//
//       The image stream is never-ending, with the ball color dependent on
//       bit depth of the current display device.  32, 24, 16 (555 and 565),
//       and 8 bit palettized types can be supplied.        
//
//       In implementation, the CSource and CSourceStream base classes from 
//       the SDK are used to implement some of the more tedious effort
//       associated with source filters.  In particular, the starting and
//       stopping of worker threads based upon overall activation/stopping 
//       is facilitated.  A worker thread sits in a loop asking for buffers
//       and then calls the PURE virtual FillBuffer method when it has a
//       buffer available to fill. 
//
//       The sample also has a simple quality management implementation in
//       the filter. With the exception of renderers (which normally initiate
//       it), this is controlled through IQualityControl.  In each frame it
//       is called for status.  Due to the straightforward nature of the 
//       filter, spacing of samples sent downward can be controlled so that
//       any CPU used runs flat out.
//
//       Demonstration instructions:
//
//       Start GraphEdit, which is available in the SDK DXUtils folder. Click
//       on the Graph menu and select "Insert Filters." From the dialog box,
//       double click on "DirectShow filters," then "Bouncing ball" and then 
//       dismiss the dialog. Go to the output pin of the filter box and 
//       right click, selecting "Render." A video renderer will be inserted 
//       and connected up (on some displays there may be a color space 
//       convertor put between them to get the pictures into a suitable 
//       format).  Then click "run" on GraphEdit and see the ball bounce 
//       around the window...
//
//       Files:
//
//       ball.cpp         Looks after drawing a moving bouncing ball
//       ball.h           Class definition for the ball drawing object
//       ball.rc          Version and title information resources
//       fball.cpp        The real filter class implementation
//       fball.h          Class definition for the main filter object
//       resource.h       A couple of identifiers for our resources
//
//       Base classes used:
//
//       CSource          Base class for a generic source filter
//       CSourceStream    A base class for a source filters stream
//
//
// Copyright (c) Microsoft Corporation.  All rights reserved.
//------------------------------------------------------------------------------


#include <streams.h>
#include "Frame.h"

namespace TheWhiteAmbit {

	//------------------------------------------------------------------------------
	// Name: CFrame::CFrame(()
	// Desc: Constructor for the ball class. The default arguments provide a
	//       reasonable image and ball size.
	//------------------------------------------------------------------------------
	CFrame::CFrame(int iImageWidth, int iImageHeight) :
m_iImageWidth(iImageWidth),
m_iImageHeight(iImageHeight)    
{

} // (Constructor)


//------------------------------------------------------------------------------
// Name: CFrame::PlotBall()
// Desc: Positions the ball on the memory buffer.
//       Assumes the image buffer is arranged as Row 1,Row 2,...,Row n
//       in memory and that the data is contiguous.
//------------------------------------------------------------------------------
void CFrame::PlotBall(BYTE pFrame[], int iPixelSize, Grid<Color>* a_pGrid)
{
	ASSERT(pFrame != NULL);

	// The current byte of interest in the frame
	BYTE *pBack;
	pBack = pFrame;

	// Plot the ball into the correct location
	BYTE *pBall = pFrame;

	for(int y = 0; y < m_iImageHeight; y++)
	{
		BYTE *pBallBase = pBall;
		for(int x = 0; x < m_iImageWidth; x++)
		{    
			D3DXVECTOR4 vecColor=a_pGrid->getPixel( m_iImageWidth - x - 1, y);
			BYTE color_r = max(0, min(255, (int)(vecColor.x*255.0f)));
			BYTE color_g = max(0, min(255, (int)(vecColor.y*255.0f)));
			BYTE color_b = max(0, min(255, (int)(vecColor.z*255.0f)));
			BYTE color_a = max(0, min(255, (int)(vecColor.w*255.0f)));

			pBallBase[0] = color_b;
			pBallBase[1] = color_g;
			pBallBase[2] = color_r;
			pBallBase += iPixelSize;
		}
		pBall += m_iImageWidth * iPixelSize;
	}
} // PlotBall

}