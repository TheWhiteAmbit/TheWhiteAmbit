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
#include "../SceneGraphLib/Node.h"
#include "../SceneGraphLib/IPresentable.h"
#include "../SceneGraphLib/PickVisitor.h"

#include "DirectX10Renderer.h"
namespace TheWhiteAmbit {
	class DirectX10ViewRender : public IPresentable
	{
		HWND m_hWindow;
		DirectX10Renderer* m_pRenderer;
		unsigned long			m_iTicks;
		bool					m_bRendererInitialized;
		Node*			m_pRootNode;
	public:
		void present(IEffect* a_pEffect);
		void setRootNode(Node* a_pRootNode);
		DirectX10ViewRender(DirectX10Renderer* a_pRenderer);
		~DirectX10ViewRender(void);
	};
}