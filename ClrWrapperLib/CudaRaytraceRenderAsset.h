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
#include "BaseNode.h"
#include "../CudaLib/CudaRaytraceRender.h"
#include "TextureAsset.h"
#include "EffectAsset.h"


namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		public ref class CudaRaytraceRenderAsset
		{
			CudaRaytraceRender*	m_pView;
		internal:
			CudaRaytraceRender* GetUnmanagedAsset(void);
		public:
			CudaRaytraceRenderAsset(DirectX9Renderer* a_pDevice);

			void Present(int effect);
			void SetTextureTarget(unsigned int a_iTextureNumber, TextureAsset^ a_pTexture);
			//void SetTextureSource(unsigned int a_iTextureNumber, TextureAsset^ a_pTexture);
			void SetRoot(BaseNode^ a_pBaseNode);
		};
	}
}