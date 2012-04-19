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

#include "../CudaLib/rendering.h"

namespace TheWhiteAmbit {

	class BSPMesh :
		public Mesh
	{
		static void splitFace(const D3DXPLANE &plane, const Face& tri, std::vector<Face>* left, std::vector<Face>* right);
		static void splitMesh(const D3DXPLANE &plane, Mesh* pMesh, BSPMesh** ppLeftMesh, BSPMesh** ppRightMesh);

		BSPMesh*	m_pLeft;
		BSPMesh*	m_pRight;
		IBisectionStrategy* m_pStrategy;
		IPickable*	m_pRootPickable;
		virtual void generateBoundingSphere(void);

		D3DXMATRIX*	m_pSplitUnitTriArray;
		D3DXPLANE*	m_pSplitPlanes;
		int*		m_pSplitIndices;
		int 		m_iNumSplits;
		int 		m_iNumSplitsUnitTri;
	public:
		BSPMesh(DWORD* a_pIndices, Vertex* a_pVertices, UINT a_iIndices, UINT a_iVertices);	
		virtual ~BSPMesh(void);
		//
		Mesh* getMesh();
		BSPMesh* getLeftMesh();
		BSPMesh* getRightMesh();
		//
		void setStrategy(IBisectionStrategy* a_pStrategy);
		//TODO: remove setPickableRoot and find cleaner way
		void setPickableRoot(IPickable* a_pPickable);
		void rekursiveSplit(void);
		void doSplit(void);

		double iterativeIntersect(Ray a_Ray, double t_in, double t_out);
		double rekursiveIntersect(Ray a_Ray, double t_in, double t_out);
		double intersectUnitTriangles(Ray a_Ray);

		//Used to generate GPU data structure
		virtual void generateSplitArrays(void);
		D3DXMATRIX*	getSplitUnitTriArray();
		D3DXPLANE*	getSplitPlanes();
		int*		getSplitIndices();
		int 		getNumSplits();
		int 		getNumSplitsUnitTri();
	};

}