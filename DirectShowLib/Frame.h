//------------------------------------------------------------------------------
// File: Ball.h
//
// Desc: DirectShow sample code - header file for the bouncing ball
//       source filter.  For more information, refer to Ball.cpp.
//
// Copyright (c) Microsoft Corporation.  All rights reserved.
//------------------------------------------------------------------------------

#pragma once

#include "../SceneGraphLib/Grid.h"
#include "../SceneGraphLib/Color.h"
#include "../DirectX9Lib/DirectX9Renderer.h"


//------------------------------------------------------------------------------
// Define GUIDS used in this sample
//------------------------------------------------------------------------------
// { fd501041-8ebe-11ce-8183-00aa00577da1 }
DEFINE_GUID(CLSID_BouncingBall, 0xfd501041, 0x8ebe, 0x11ce, 0x81, 0x83, 0x00, 0xaa, 0x00, 0x57, 0x7d, 0xa1);


//------------------------------------------------------------------------------
// Class CFrame
//
// This class encapsulates the behavior of the bounching ball over time
//------------------------------------------------------------------------------
namespace TheWhiteAmbit {
	class CFrame
	{
	public:

		CFrame(int iImageWidth = 1280, int iImageHeight = 1024);

		// Plots the square ball in the image buffer, at the current location.
		// Use BallPixel[] as pixel value for the ball.
		// Plots zero in all 'background' image locations.
		// iPixelSize - the number of bytes in a pixel (size of BallPixel[])
		void PlotBall(BYTE pFrame[], int iPixelSize, Grid<Color>* a_pGrid);


		int GetImageWidth() { return m_iImageWidth ;}
		int GetImageHeight() { return m_iImageHeight ;}

	private:

		// The dimensions we can plot in, allowing for the width of the ball

		int m_iImageHeight;     // The image height
		int m_iImageWidth;      // The image width

	}; // CFrame
}