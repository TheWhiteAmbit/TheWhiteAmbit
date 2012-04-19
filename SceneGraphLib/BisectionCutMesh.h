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

#include <vector>
#include "Mesh.h"
#include "Face.h"
#include "IPickable.h"
#include "IBisectionStrategy.h"

namespace TheWhiteAmbit {

	class BisectionCutMesh :
		public Mesh
	{
		static bool cutFace1and2(const D3DXPLANE &plane, const Face& tri, Face* t_out);
		static bool cutFace1and1(const D3DXPLANE &plane, const Face& tri, Face* t_out);
		static void cutFace(const D3DXPLANE &plane, const Face& tri, std::vector<Face>* left, std::vector<Face>* right);

		BisectionCutMesh*	m_pLeft;
		BisectionCutMesh*	m_pRight;
		IBisectionStrategy* m_pStrategy;
		IPickable*	m_pRootPickable;
		D3DXMATRIX*	m_pUnitTriangles;
		unsigned int m_iNumUnitTriangles;
	public:
		BisectionCutMesh(DWORD* a_pIndices, Vertex* a_pVertices, UINT a_iIndices, UINT a_iVertices);	
		virtual ~BisectionCutMesh(void);

		static void cutMesh(const D3DXPLANE &plane, Mesh* pMesh, BisectionCutMesh** ppLeftMesh, BisectionCutMesh** ppRightMesh);
		//
		Mesh* getMesh();
		BisectionCutMesh* getLeftMesh();
		BisectionCutMesh* getRightMesh();
		//
		void setStrategy(IBisectionStrategy* a_pStrategy);
		void setPickableRoot(IPickable* a_pPickable);
		void doSplit(void);
		void rekursiveSplit(void);
	};
}

