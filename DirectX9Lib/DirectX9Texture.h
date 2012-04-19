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

#include "DirectX9Renderer.h"
#include "../SceneGraphLib/Grid.h"
#include "../SceneGraphLib/Color.h"

namespace TheWhiteAmbit {

	class DirectX9Texture
	{
		IDirect3DTexture9*	m_pTexture;
		DirectX9Renderer*	m_pRenderer;

		unsigned int m_iWidth;
		unsigned int m_iHeight;
	public:
		DirectX9Texture(DirectX9Renderer* a_pRenderer);
		DirectX9Texture(DirectX9Renderer* a_pRenderer, unsigned int width, unsigned int height, DWORD usage, D3DFORMAT a_hFormat);
		virtual ~DirectX9Texture(void);

		IDirect3DTexture9* getTexture(void);
		IDirect3DSurface9* getSurface(unsigned int level);
		Grid<Color>* getGrid(void);
		void setGrid(Grid<Color>* a_pGrid);
		void setGrid(Grid<unsigned char[4]>* a_pGrid);

		unsigned int getWidth(void);
		unsigned int getHeight(void);

		void fillColorGardient(void);
	};

}