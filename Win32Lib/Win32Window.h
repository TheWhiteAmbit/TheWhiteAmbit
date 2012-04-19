//The White Ambit, Rendering-Framework
//Copyright (C) 2009  Moritz Strickhausen

//This program is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with this program.  If not, see <http://www.gnu.org/licenses/>.



#pragma once
#include <windows.h>
#include "IWin32WindowProc.h"


class Win32Window
	:	public IWin32WindowProc
{
	WNDCLASSEX	m_WindowClassEx;
	HINSTANCE   m_hInstance;
	HWND        m_hWindow;
	PAINTSTRUCT m_PaintStruct;
	HDC			m_hDeviceContext;
	bool		m_bWindowLoopRunning;
public:
	//--------------------------------------------------------------------------------------
	// Register class and create window
	//--------------------------------------------------------------------------------------
	HRESULT InitWindow( HINSTANCE a_hInstance, int a_nCmdShow );
	HWND getWindowHandle(void);
	bool getWindowLoopRunning(void);

	//--------------------------------------------------------------------------------------
	// Called every time the application receives a message
	//--------------------------------------------------------------------------------------
	static LRESULT CALLBACK WndProc( HWND a_hWindow, UINT a_Message, WPARAM a_WParam, LPARAM a_LParam );
	virtual LRESULT CALLBACK windowProc( HWND a_hWindow, UINT a_Message, WPARAM a_WParam, LPARAM a_LParam );
	Win32Window(void);
	virtual ~Win32Window(void);
};
