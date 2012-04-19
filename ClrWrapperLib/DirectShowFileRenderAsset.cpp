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
#include "DirectShowFileRenderAsset.h"

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		DirectShowFileRenderAsset::DirectShowFileRenderAsset(System::String^ a_sFilename)
		{
			const wchar_t* str=(const wchar_t*)(System::Runtime::InteropServices::Marshal::StringToHGlobalUni(a_sFilename).ToPointer());
			this->m_pView=new TheWhiteAmbit::DirectShowFileRender(str);		
		}

		DirectShowFileRender* DirectShowFileRenderAsset::GetUnmanagedAsset(void)
		{
			return this->m_pView;
		}

		void DirectShowFileRenderAsset::Present(int effect)
		{
			this->m_pView->present(effect);
		}

		void DirectShowFileRenderAsset::SetFrameBuffer(GridAsset^ a_pGrid)
		{
			this->m_pView->setFrameBuffer(a_pGrid->GetUnmanagedAsset());
		}

		void DirectShowFileRenderAsset::Release(){
			delete this->m_pView;
			this->m_pView=NULL;
		}
	}
}