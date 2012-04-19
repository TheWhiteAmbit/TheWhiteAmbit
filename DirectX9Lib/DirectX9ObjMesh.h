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

//#include <windows.h>
//#include <d3d9.h>
//#include <d3dx9.h>
#include "DirectX9Renderer.h"
//#include "DirectX9Effect.h"

#include "DirectX9ObjMeshLoader.h"

namespace TheWhiteAmbit {

	class DirectX9ObjMesh
	{
		CMeshLoader                 m_MeshLoader;            // Loads a mesh from an .obj file
		ID3DXEffect* m_pEffect;
		void RenderSubset( UINT iSubset );
	public:
		ID3DXMesh* getMesh(void);
		void Render( ID3DXEffect* a_pEffect );
		DirectX9ObjMesh(DirectX9Renderer* a_pRenderer, LPCWSTR a_sFilename);
		~DirectX9ObjMesh(void);
	};

}