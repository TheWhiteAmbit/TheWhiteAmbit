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
//#include <d3d9.h>
//#include <d3dx9.h>

#include "../SceneGraphLib/Node.h"
#include "../SceneGraphLib/IRenderable.h"
#include "DirectX9Renderer.h"
namespace TheWhiteAmbit {
	class DirectX9ClearBuffer :
		public Node, IRenderable
	{
		DirectX9Renderer*	m_pRenderer;
	public:
		virtual void render(IEffect* a_pEffect);
		virtual void accept(RenderVisitor* a_pRenderVisitor);
		DirectX9ClearBuffer(DirectX9Renderer* a_pRenderer);
		virtual ~DirectX9ClearBuffer(void);
	};
}
