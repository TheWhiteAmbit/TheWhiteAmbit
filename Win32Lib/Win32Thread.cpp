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



#include "Win32Thread.h"

Win32Thread::Win32Thread(IRunnable* a_pRunable){
	m_pRunable=a_pRunable;
}

Win32Thread::~Win32Thread(void)
{
}

unsigned __stdcall Win32Thread::ThreadFunction(void *ArgList) {
	IRunnable* pRunable	= (IRunnable*)ArgList;
	unsigned int exitCode=pRunable->run();
#ifdef NGN_USES_MSVCRT
	_endthreadex(exitCode);
#endif
	return exitCode;
}

void Win32Thread::start(){
#ifndef NGN_USES_MSVCRT
	m_hThread = (uintptr_t)CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)ThreadFunction, (void*)m_pRunable, 0, (LPDWORD)&m_uiThreadId);
#else
	m_hThread = _beginthreadex(NULL, 0, ThreadFunction, (void*)m_pRunable, 0, &m_uiThreadId);
#endif 
}

void Win32Thread::kill(){
	DWORD dwExitCode=0;
	TerminateThread((HANDLE)m_hThread, dwExitCode);
	CloseHandle((HANDLE)m_hThread);
}

unsigned int Win32Thread::join(){
	//TODO: look if the TRUE for APC is needed
	WaitForSingleObjectEx((HANDLE)m_hThread, INFINITE, TRUE);
	DWORD exitCode=0;
	GetExitCodeThread((HANDLE)m_hThread, &exitCode);
	CloseHandle((HANDLE)m_hThread);
	return exitCode;
}

void Win32Thread::pause(){
	SuspendThread((HANDLE)m_hThread);
}

void Win32Thread::resume(){
	ResumeThread((HANDLE)m_hThread);
}