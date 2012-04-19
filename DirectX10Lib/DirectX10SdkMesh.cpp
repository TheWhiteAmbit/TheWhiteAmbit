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




#pragma comment (lib, "DirectXUTLib.lib")

#include "DirectX10SdkMesh.h"
namespace TheWhiteAmbit {
	DirectX10SdkMesh::DirectX10SdkMesh(DirectX10Renderer* a_pRenderer, LPCWSTR a_sFilename)
	{
		m_pMesh=new CDXUTSDKMesh();
		m_pMesh->Create(a_pRenderer->getDevice(), a_sFilename, true);
	}

	DirectX10SdkMesh::~DirectX10SdkMesh(void)
	{
		if(m_pMesh)
		{
			m_pMesh->Destroy();
			delete m_pMesh;
		}
	}

	CDXUTSDKMesh * DirectX10SdkMesh::getMesh(void)
	{
		return m_pMesh;
	}
}