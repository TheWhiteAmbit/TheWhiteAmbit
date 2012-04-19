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



#include "RaytraceVisitor.h"
#include "Sphere.h"
#include "Tetrahedron.h"

namespace TheWhiteAmbit {

	RaytraceVisitor::RaytraceVisitor(void) : PickVisitor(D3DXVECTOR2(0, 0))
	{
		m_pMatrices=NULL;
		m_pBSPMesh=NULL;
	}

	RaytraceVisitor::~RaytraceVisitor(void)
	{
		if(m_pMatrices)
			delete m_pMatrices;
	}

	void RaytraceVisitor::visit(IRaytraceable* a_pSceneNode)
	{
		if(!m_pMatrices)
		{
			m_pMatrices=new D3DXMATRIX[24];		
			m_pMatrices[0]=a_pSceneNode->getWorld();
			m_pMatrices[1]=a_pSceneNode->getView();
			m_pMatrices[2]=a_pSceneNode->getProjection();
			D3DXMatrixMultiply(&m_pMatrices[3], &a_pSceneNode->getWorld(), &a_pSceneNode->getView());
			D3DXMatrixMultiply(&m_pMatrices[4], &a_pSceneNode->getView(), &a_pSceneNode->getProjection());
			D3DXMatrixMultiply(&m_pMatrices[5], &m_pMatrices[3], &a_pSceneNode->getProjection());
			D3DXMatrixInverse(&m_pMatrices[6], NULL, &m_pMatrices[0]);
			D3DXMatrixInverse(&m_pMatrices[7], NULL, &m_pMatrices[1]);
			D3DXMatrixInverse(&m_pMatrices[8], NULL, &m_pMatrices[2]);
			D3DXMatrixInverse(&m_pMatrices[9], NULL, &m_pMatrices[3]);
			D3DXMatrixInverse(&m_pMatrices[10], NULL, &m_pMatrices[4]);
			D3DXMatrixInverse(&m_pMatrices[11], NULL, &m_pMatrices[5]);

			m_pMatrices[12+0]=a_pSceneNode->getLastWorld();
			m_pMatrices[12+1]=a_pSceneNode->getLastView();
			m_pMatrices[12+2]=a_pSceneNode->getLastProjection();
			D3DXMatrixMultiply(&m_pMatrices[12+3], &a_pSceneNode->getLastWorld(), &a_pSceneNode->getLastView());
			D3DXMatrixMultiply(&m_pMatrices[12+4], &a_pSceneNode->getLastView(), &a_pSceneNode->getLastProjection());
			D3DXMatrixMultiply(&m_pMatrices[12+5], &m_pMatrices[12+3], &a_pSceneNode->getLastProjection());
			D3DXMatrixInverse(&m_pMatrices[12+6], NULL, &m_pMatrices[12+0]);
			D3DXMatrixInverse(&m_pMatrices[12+7], NULL, &m_pMatrices[12+1]);
			D3DXMatrixInverse(&m_pMatrices[12+8], NULL, &m_pMatrices[12+2]);
			D3DXMatrixInverse(&m_pMatrices[12+9], NULL, &m_pMatrices[12+3]);
			D3DXMatrixInverse(&m_pMatrices[12+10], NULL, &m_pMatrices[12+4]);
			D3DXMatrixInverse(&m_pMatrices[12+11], NULL, &m_pMatrices[12+5]);

			m_pBSPMesh=a_pSceneNode->getBSPMesh();
		}
	}

	D3DXMATRIX*		RaytraceVisitor::getMatrices(void)
	{
		return m_pMatrices;
	}

	BSPMesh* RaytraceVisitor::getBSPMesh()
	{
		return m_pBSPMesh;
	}
}