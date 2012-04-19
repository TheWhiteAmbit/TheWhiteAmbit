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




#include "WindowMessageVisitor.h"
namespace TheWhiteAmbit {
	WindowMessageVisitor::WindowMessageVisitor(HWND hWindow, UINT message, WPARAM	wParam, LPARAM lParam)
	{
		m_hWindow=hWindow;
		m_Message=message;
		m_WParam=wParam;
		m_LParam=lParam;
	}

	WindowMessageVisitor::~WindowMessageVisitor(void)
	{
	}

	void WindowMessageVisitor::visit(IWin32WindowProc* a_pSceneNode)
	{
		a_pSceneNode->windowProc(m_hWindow, m_Message, m_WParam, m_LParam);
	}
}