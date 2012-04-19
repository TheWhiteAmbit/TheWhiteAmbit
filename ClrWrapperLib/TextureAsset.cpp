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



#include "stdafx.h"

#include "TextureAsset.h"

using namespace System;

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		DirectX9Texture* TextureAsset::GetUnmanagedAsset(void)
		{
			return this->m_pTexture;
		}

		TextureAsset::TextureAsset(DirectX9Renderer* a_pDevice)
		{
			this->m_pTexture=new DirectX9Texture(a_pDevice);
		}

		TextureAsset::TextureAsset(DirectX9Texture* a_pTexture)
		{
			this->m_pTexture=a_pTexture;
		}
	}
}