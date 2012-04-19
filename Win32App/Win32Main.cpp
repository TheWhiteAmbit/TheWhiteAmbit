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



//--------------------------------------------------------------------------------------
// File: Win32Main.cpp
//
// This tutorial sets up a window with a message pump, and a renderer
//--------------------------------------------------------------------------------------
#include "Win32Main.h"

#pragma comment (lib, "Win32Lib.lib")
#include "../Win32Lib/Win32Thread.h"
#include "../Win32Lib/Win32WindowRunnable.h"

#include <iostream>
#include "../DirectXUTLib/DXUT.h"
#pragma comment (lib, "DirectXUTLib.lib")

#pragma comment (lib, "DirectX9Lib.lib")
#include "../DirectX9Lib/DirectX9ViewRender.h"
#include "../DirectX9Lib/DirectX9DrawMock.h"

#pragma comment (lib, "CudaLib.lib")
#include "Scene1.h"


//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
int WINAPI wWinMain( HINSTANCE a_hInstance, HINSTANCE a_hPrevInstance, LPWSTR a_lpCmdLine, int a_nCmdShow )
{
	Win32WindowRunnable* runnableWindow=new Win32WindowRunnable(a_hInstance, a_nCmdShow);
	IThread* threadWindow=new Win32Thread(runnableWindow);
	threadWindow->start();
	

	Scene1* scene=new Scene1(runnableWindow->waitForWindowHandle());
	IThread* threadRender=new Win32Thread(scene);
	threadRender->start();
	
	
	threadWindow->join();
	threadRender->pause();
	threadRender->kill();
	//threadRender->join();

	//delete threadWindow;
	//delete threadRender;
	delete runnableWindow;
	//delete runnableRender;

	return 0;
}