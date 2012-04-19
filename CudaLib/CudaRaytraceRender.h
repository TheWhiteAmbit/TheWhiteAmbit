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
#include "cudatexturerender.h"
#include "..\SceneGraphLib\Node.h"
#include "..\SceneGraphLib\BSPMesh.h"
#include "..\SceneGraphLib\RaytraceVisitor.h"
namespace TheWhiteAmbit {
	class CudaRaytraceRender :
		public CudaTextureRender
	{
		Node*				m_pRootNode;
		float*				m_pUnitTriangles;
		float*				m_pProjectionMatrices;
		float*				m_pSplitPlanes;
		int*				m_pSplitIndices;
		bool				m_bInitDone;
		BSPMesh*			m_pBSPMesh;
		virtual void RunKernelsEffect(int method);
	public:
		CudaRaytraceRender(DirectX9Renderer* a_pRenderer);
		CudaRaytraceRender(DirectX10Renderer* a_pRenderer);
		~CudaRaytraceRender(void);
		void setBSPMesh(BSPMesh* a_pBSPMesh);
		virtual void present(int effect);
		void setRootNode(Node* a_pNode);
		//virtual void setTextureSource(unsigned int a_iTextureNumber, DirectX9Texture* a_pTexture);
		//virtual void setTextureSource(unsigned int a_iTextureNumber, DirectX10Texture* a_pTexture);
	};
}