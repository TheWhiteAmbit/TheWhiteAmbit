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
#include "Transform.h"
//#include "WindowMessageVisitor.h"
#include "..\Win32Lib\IWin32WindowProc.h"
#include "..\DirectXUTLib\DXUT.h"
#include "..\DirectXUTLib\DXUTcamera.h"
namespace TheWhiteAmbit {
	class Trackball :
		public Transform, IWin32WindowProc
	{
		//CModelViewerCamera* m_pCamera;
		CFirstPersonCamera* m_pCamera;
	public:
		Trackball(void);
		virtual ~Trackball(void);
		virtual LRESULT CALLBACK windowProc( HWND a_hWindow, UINT a_Message, WPARAM a_WParam, LPARAM a_LParam );
		virtual void accept(WindowMessageVisitor* a_pWindowMessageVisitor);
	};
}
