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
#include "IRunnable.h"
#include "Win32Window.h"

class Win32WindowRunnable :
	public IRunnable
{
	HINSTANCE	m_hInstance;
	int			m_nCmdShow;
	Win32Window* m_pWindow;
public:

	virtual unsigned int run();
	HWND waitForWindowHandle(void);
	Win32WindowRunnable(HINSTANCE a_hInstance, int a_nCmdShow);
	virtual ~Win32WindowRunnable(void);
};
