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
#include "RaytraceRenderAsset.h"

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		RaytraceRenderAsset::RaytraceRenderAsset(void)
		{
			this->m_pView=new RaytraceRender((unsigned int)RESOLUTION_X, (unsigned int)RESOLUTION_Y);
		}

		RaytraceRender* RaytraceRenderAsset::GetUnmanagedAsset(void)
		{
			return this->m_pView;
		}

		void RaytraceRenderAsset::SetRoot(BaseNode^ a_pBaseNode)
		{
			m_pView->setRootNode(a_pBaseNode->GetUnmanagedNode());
		}

		void RaytraceRenderAsset::Present(EffectAsset^ a_pEffect)
		{
			m_pView->present();
		}
	}
}