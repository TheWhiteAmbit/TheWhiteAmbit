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

#include "DirectX10Renderer.h"
#include "../SceneGraphLib/Grid.h"
#include "../SceneGraphLib/Color.h"

namespace TheWhiteAmbit {
class DirectX10Texture
{
	DirectX10Renderer* m_pRenderer;
	ID3D10Texture2D* m_pTexture;
public:
	DirectX10Texture(DirectX10Renderer* a_pRenderer);
	DirectX10Texture(DirectX10Renderer* a_pRenderer, unsigned int width, unsigned int height, D3D10_USAGE usage, DXGI_FORMAT format, UINT bindFlags);
	virtual ~DirectX10Texture(void);

	ID3D10Texture2D* getTexture(void);
	Grid<Color>* getGrid(void);
	void setGrid(Grid<Color>* a_pGrid);

	unsigned int getWidth(void);
	unsigned int getHeight(void);
	
	void fillColorGardient(void);
};
}