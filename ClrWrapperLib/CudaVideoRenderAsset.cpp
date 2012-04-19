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



#include "StdAfx.h"
#include "CudaVideoRenderAsset.h"

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		CudaVideoRenderAsset::CudaVideoRenderAsset(DirectX9Renderer* a_pDevice, System::String^ a_sFilename)
		{
			const wchar_t* str=(const wchar_t*)(System::Runtime::InteropServices::Marshal::StringToHGlobalUni(a_sFilename).ToPointer());
			this->m_pView=new CudaVideoRender(a_pDevice, str);		
		}	

		CudaVideoRender* CudaVideoRenderAsset::GetUnmanagedAsset(void)
		{
			return this->m_pView;
		}

		bool CudaVideoRenderAsset::HasNext(){		
			return this->m_pView->hasNext();		
		}

		void CudaVideoRenderAsset::Present(int effect)
		{
			this->m_pView->present(effect);
		}

		void CudaVideoRenderAsset::SetTextureTarget(unsigned int a_iTextureNumber, TextureAsset^ a_pTexture)
		{
			this->m_pView->setTextureTarget(a_iTextureNumber, a_pTexture->GetUnmanagedAsset());
		}

		void CudaVideoRenderAsset::Release()
		{
			//TODO: make IDisposable and so on, maybe also use smartpointers for unmanages resources like textures etc.
			this->m_pView->release();
		}
	}
}