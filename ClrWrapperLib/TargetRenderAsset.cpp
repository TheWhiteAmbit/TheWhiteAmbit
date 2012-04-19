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

#include "TargetRenderAsset.h"

using namespace System;

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		DirectX9TargetRender* TargetRenderAsset::GetUnmanagedAsset(void)
		{
			return this->m_pView;
		}

		TargetRenderAsset::TargetRenderAsset(DirectX9Renderer* a_pDevice)
		{
			this->m_pView=new DirectX9TargetRender(a_pDevice);
		}

		TargetRenderAsset::TargetRenderAsset(DirectX9Renderer* a_pDevice, D3DFORMAT a_D3dFormat)
		{
			this->m_pView=new DirectX9TargetRender(a_pDevice, a_D3dFormat);
		}

		void TargetRenderAsset::SetRoot(BaseNode^ a_pBaseNode)
		{
			m_pView->setRootNode(a_pBaseNode->GetUnmanagedNode());
		}

		System::IntPtr^ TargetRenderAsset::GetDirect3D9Surface(unsigned int level){
			return gcnew System::IntPtr(m_pView->getTexture()->getSurface(level));
		}

		void TargetRenderAsset::Present(EffectAsset^ a_pEffect)
		{
			m_pView->present(a_pEffect->GetUnmanagedAsset());
		}	
	}
}