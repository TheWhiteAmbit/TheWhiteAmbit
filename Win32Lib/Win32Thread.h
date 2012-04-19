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
#include <process.h>

#include "IThread.h"
#include "IRunnable.h"

class Win32Thread :
	public IThread
{
	IRunnable* m_pRunable;
	unsigned int m_uiThreadId;
	uintptr_t m_hThread;
public:
	Win32Thread(IRunnable* a_pRunable);
	virtual ~Win32Thread();
	static unsigned __stdcall ThreadFunction(void *ArgList);
	virtual void start();
	virtual void kill();
	virtual unsigned int join();
	virtual void pause();
	virtual void resume();	
};
