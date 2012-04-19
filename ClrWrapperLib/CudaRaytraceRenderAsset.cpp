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
#include "CudaRaytraceRenderAsset.h"

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		CudaRaytraceRenderAsset::CudaRaytraceRenderAsset(DirectX9Renderer* a_pDevice)
		{
			this->m_pView=new CudaRaytraceRender(a_pDevice);
		}

		CudaRaytraceRender* CudaRaytraceRenderAsset::GetUnmanagedAsset(void)
		{
			return this->m_pView;
		}

		void CudaRaytraceRenderAsset::Present(int effect)
		{
			this->m_pView->present(effect);
		}

		void CudaRaytraceRenderAsset::SetTextureTarget(unsigned int a_iTextureNumber, TextureAsset^ a_pTexture)
		{
			this->m_pView->setTextureTarget(a_iTextureNumber, a_pTexture->GetUnmanagedAsset());
		}

		//void CudaRaytraceRenderAsset::SetTextureSource(unsigned int a_iTextureNumber, TextureAsset^ a_pTexture)
		//{
		//	this->m_pView->setTextureSource(a_iTextureNumber, a_pTexture->GetUnmanagedAsset());
		//}

		void CudaRaytraceRenderAsset::SetRoot(BaseNode^ a_pBaseNode)
		{
			m_pView->setRootNode(a_pBaseNode->GetUnmanagedNode());
		}
	}
}