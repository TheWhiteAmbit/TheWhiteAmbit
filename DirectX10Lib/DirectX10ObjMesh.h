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

#include "DirectX10Renderer.h"
#include "DirectX10ObjMeshLoader.h"
namespace TheWhiteAmbit {
	class DirectX10ObjMesh
	{
		CMeshLoader10                 m_MeshLoader;            // Loads a mesh from an .obj file
		ID3D10Effect*				  m_pEffect;
		void RenderSubset( UINT iSubset );
	public:
		void Render( ID3D10Effect* a_pEffect );
		DirectX10ObjMesh(DirectX10Renderer* a_pRenderer, LPCWSTR a_sFilename);
		ID3DX10Mesh* getMesh();
		~DirectX10ObjMesh(void);
	};
}