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

#include "SdkMeshAsset.h"

using namespace System;

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		DirectX9SdkMesh* SdkMeshAsset::GetUnmanagedAsset(void)
		{
			return this->m_pMesh;
		}

		SdkMeshAsset::SdkMeshAsset(DirectX9Renderer* a_pDevice, System::String^ a)
		{
			const wchar_t* str=(const wchar_t*)(System::Runtime::InteropServices::Marshal::StringToHGlobalUni(a).ToPointer());
			m_pMesh=new DirectX9SdkMesh(a_pDevice, str);
		}
	}
}