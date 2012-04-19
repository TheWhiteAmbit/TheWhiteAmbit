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
#include "../DirectX9Lib/DirectX9Texture.h"

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		public ref class TextureAsset
		{
			DirectX9Texture* m_pTexture;
		internal:		
			DirectX9Texture* GetUnmanagedAsset(void);
		public:
			TextureAsset(DirectX9Texture* a_pTexture);
			TextureAsset(DirectX9Renderer* a_pDevice);
		};
	}
}