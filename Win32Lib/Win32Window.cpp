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



#include "Win32Window.h"
#include "../CudaLib/rendering.h"

#ifndef GWL_USERDATA
#define GWL_USERDATA (-21)
#endif

Win32Window::Win32Window(void)
{
	m_bWindowLoopRunning=true;
	m_hWindow=NULL;
}

Win32Window::~Win32Window(void)
{
	SetWindowLongPtr (m_hWindow, GWL_USERDATA, (LONG)NULL);
}

HRESULT Win32Window::InitWindow( HINSTANCE a_hInstance, int a_nCmdShow )
{
	// Register class

	m_WindowClassEx.cbSize = sizeof( WNDCLASSEX );
	m_WindowClassEx.style = CS_HREDRAW | CS_VREDRAW;
	m_WindowClassEx.lpfnWndProc = WndProc;
	m_WindowClassEx.cbClsExtra = 0;
	m_WindowClassEx.cbWndExtra = 0;
	m_WindowClassEx.hInstance = a_hInstance;
	m_WindowClassEx.hIcon = LoadIcon( a_hInstance, ( LPCTSTR )IDI_WINLOGO );
	m_WindowClassEx.hCursor = LoadCursor( NULL, IDC_ARROW );
	//m_WindowClassEx.hbrBackground = ( HBRUSH )( COLOR_WINDOW + 1 );
	//m_WindowClassEx.hbrBackground = ( HBRUSH )( COLOR_BACKGROUND );
	m_WindowClassEx.hbrBackground = 0;
	m_WindowClassEx.lpszMenuName = NULL;
	m_WindowClassEx.lpszClassName = L"Win32WindowClass";
	m_WindowClassEx.hIconSm = LoadIcon( m_WindowClassEx.hInstance, ( LPCTSTR )IDI_WINLOGO );
	if( !RegisterClassEx( &m_WindowClassEx ) )
		return E_FAIL;

	// Create window
	m_hInstance = a_hInstance;
	RECT rc = { 0, 0, (LONG)RESOLUTION_X, (LONG)RESOLUTION_Y };
	AdjustWindowRect( &rc, WS_OVERLAPPEDWINDOW, FALSE );
	m_hWindow = CreateWindow( L"Win32WindowClass", L"SimpleWindow", WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top, NULL, NULL, a_hInstance,
		this );

	if( !m_hWindow )
		return E_FAIL;

	IWin32WindowProc* pWindow = (IWin32WindowProc*) GetWindowLongPtr (m_hWindow, GWL_USERDATA) ;	
	if(this!=pWindow)
		return E_FAIL;

	ShowWindow( m_hWindow, a_nCmdShow );

	return S_OK;
}

HWND Win32Window::getWindowHandle(void)
{
	return this->m_hWindow;
}

bool Win32Window::getWindowLoopRunning(void)
{
	return m_bWindowLoopRunning;
}


//--------------------------------------------------------------------------------------
// Called every time the application receives a message
//--------------------------------------------------------------------------------------
LRESULT CALLBACK Win32Window::WndProc( HWND a_hWindow, UINT a_Message, WPARAM a_WParam, LPARAM a_LParam )
{
	IWin32WindowProc* pWindow = NULL;
	switch(a_Message)
	{
	case WM_NCCREATE:
		{
			pWindow = (IWin32WindowProc*) ((LPCREATESTRUCT) a_LParam)->lpCreateParams ;
			SetWindowLongPtr (a_hWindow, GWL_USERDATA, (LONG)pWindow) ;
		}
		break;
	}

	pWindow = (IWin32WindowProc*) GetWindowLongPtr (a_hWindow, GWL_USERDATA) ;	
	if(pWindow)
		return pWindow->windowProc(a_hWindow, a_Message, a_WParam, a_LParam);
	else
		return DefWindowProc (a_hWindow, a_Message, a_WParam, a_LParam) ;
}

LRESULT CALLBACK Win32Window::windowProc( HWND a_hWindow, UINT a_Message, WPARAM a_WParam, LPARAM a_LParam )
{
	switch( a_Message )
	{
	case WM_PAINT:
		{
			//m_hDeviceContext = BeginPaint( a_hWindow, &m_PaintStruct );
			//EndPaint( a_hWindow, &m_PaintStruct );
			Sleep(0);
		}
		break;
	case WM_DESTROY:
		{
			m_bWindowLoopRunning=false;
			PostQuitMessage( 0 );
		}
		break;
	default:
		return DefWindowProc( a_hWindow, a_Message, a_WParam, a_LParam );
	}
	return 0;	//message was handled by some switch case
}