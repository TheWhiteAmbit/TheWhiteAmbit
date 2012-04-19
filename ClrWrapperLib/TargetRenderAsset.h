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
#include "../DirectX9Lib/DirectX9TargetRender.h"
#include "BaseNode.h"
#include "EffectAsset.h"
#include "CameraAsset.h"

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		public ref class TargetRenderAsset
		{
			DirectX9TargetRender* m_pView;
		internal:
			DirectX9TargetRender* GetUnmanagedAsset(void);
		public:

			System::IntPtr^ GetDirect3D9Surface(unsigned int level);
			void		  SetRoot(BaseNode^ a_pBaseNode);
			void		  Present(EffectAsset^ a_pEffect);
			TargetRenderAsset(DirectX9Renderer* a_pDevice);
			TargetRenderAsset(DirectX9Renderer* a_pDevice, D3DFORMAT a_D3dFormat);
		};
	}
}