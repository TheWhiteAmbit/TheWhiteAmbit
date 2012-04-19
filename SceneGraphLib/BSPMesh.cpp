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




#include "BSPMesh.h"
#include "BisectionCutMesh.h"
#include "PickVisitor.h"
#include <limits>

#include <map>

namespace TheWhiteAmbit {

	BSPMesh::BSPMesh(DWORD* a_pIndices, Vertex* a_pVertices, UINT a_iIndices, UINT a_iVertices)
		: Mesh(a_pIndices, a_pVertices, a_iIndices, a_iVertices)
	{
		m_pLeft=NULL;
		m_pRight=NULL;
		m_pStrategy=NULL;
		m_pRootPickable=NULL;

		m_pSplitUnitTriArray=NULL;
		m_iNumUnitTriangles=0;
		m_pSplitPlanes=NULL;
		m_pSplitIndices=NULL;
		m_iNumSplits=0;
		m_iNumSplitsUnitTri=0;
	}

	BSPMesh::~BSPMesh(void)
	{
	}

	void BSPMesh::setStrategy(IBisectionStrategy* a_pStrategy)
	{
		this->m_pStrategy=a_pStrategy;
	}

	void BSPMesh::setPickableRoot(IPickable* a_pPickable)
	{
		m_pRootPickable=a_pPickable;
	}

	double BSPMesh::intersectUnitTriangles(Ray a_Ray)
	{
		//TODO: move generateUnitTriangles to a better place
		if(!this->m_pUnitTriangles)
			this->generateUnitTriangles();
		double result=std::numeric_limits<double>::infinity();
		D3DXVECTOR4 vecOrig;
		D3DXVECTOR4 vecDir;
		D3DXMATRIX* m;
		for(unsigned int i=0;i<this->m_iNumUnitTriangles;i++)
		{
			m=&this->m_pUnitTriangles[i];

			D3DXVec4Transform(&vecOrig, &a_Ray.orig, m);
			D3DXVec4Transform(&vecDir, &a_Ray.dir, m);

			float t=-vecOrig.z/vecDir.z;
			float u=vecOrig.x+t*vecDir.x;
			float v=vecOrig.y+t*vecDir.y;

			if(v<0.0f || u<0.0f || u+v>1.0f || vecDir.z<0.0f)
				continue;
			else
				result=min(result, (double)t);
		}
		return result;
	}

	void BSPMesh::generateSplitArrays(void)
	{
		if(!m_pSplitUnitTriArray||!m_pSplitPlanes||!m_pSplitIndices)
		{
			if(m_pSplitUnitTriArray)
				delete m_pSplitUnitTriArray;
			if(m_pSplitPlanes)
				delete m_pSplitPlanes;
			if(m_pSplitIndices)
				delete m_pSplitIndices;
			m_pSplitUnitTriArray=NULL;
			m_pSplitPlanes=NULL;
			m_pSplitIndices=NULL;
			m_iNumSplits=0;
			m_iNumSplitsUnitTri=0;


			std::vector<D3DXMATRIX*>*	listSplitUnitTriArray=new std::vector<D3DXMATRIX*>();
			std::vector<int>*			listSplitUnitTriCount=new std::vector<int>();
			std::vector<D3DXPLANE>*		listSplitPlanes=new std::vector<D3DXPLANE>();
			std::vector<int>*			listSplitIndices=new std::vector<int>();
			std::map<int, BSPMesh*>* mapIndexNode=new std::map<int, BSPMesh*>();
			std::map<BSPMesh*, int>* mapNodeIndex=new std::map<BSPMesh*, int>();	

			BSPMesh* splitList[128];
			unsigned int index=0;
			splitList[index]=this;
			index=1;
			m_iNumSplits=0;
			while(index)
			{	
				index--;
				BSPMesh* current=splitList[index];
				mapIndexNode->insert(std::pair<int, BSPMesh*>(m_iNumSplits, current));
				mapNodeIndex->insert(std::pair<BSPMesh*, int>(current, m_iNumSplits));
				m_iNumSplits++;

				if(current->m_pRight)
				{
					splitList[index]=current->m_pRight;
					index++;		
				}
				if(current->m_pLeft){
					splitList[index]=current->m_pLeft;
					index++;
				}
			}

			m_iNumSplitsUnitTri=0;
			for(int i=0; i<m_iNumSplits; i++)
			{
				BSPMesh* current=mapIndexNode->find(i)->second;
				int leftIndex=0;
				if(current->m_pLeft)
					leftIndex=mapNodeIndex->find(current->m_pLeft)->second;
				int rightIndex=0;
				if(current->m_pRight)
					rightIndex=mapNodeIndex->find(current->m_pRight)->second;

				int triangleCount=0;
				if(!current->m_pLeft && !current->m_pRight)
				{
					triangleCount=current->getNumUnitTriangles();
					listSplitUnitTriCount->push_back(triangleCount);
					listSplitUnitTriArray->push_back(current->getUnitTriangles());
				}

				listSplitPlanes->push_back(current->m_pStrategy->getSplitPlane());
				listSplitIndices->push_back(triangleCount);
				listSplitIndices->push_back(leftIndex);
				listSplitIndices->push_back(rightIndex);
				listSplitIndices->push_back(m_iNumSplitsUnitTri);

				if(!current->m_pLeft && !current->m_pRight)
					m_iNumSplitsUnitTri++;
			}

			m_pSplitUnitTriArray=new D3DXMATRIX[m_iNumSplitsUnitTri*UNIT_TRIANGLE_MAX];
			m_pSplitPlanes=new D3DXPLANE[m_iNumSplits];
			m_pSplitIndices=new int[m_iNumSplits*4];

			for(int i=0;i<m_iNumSplits; i++)
			{
				m_pSplitPlanes[i]=listSplitPlanes->at(i);
				m_pSplitIndices[i*4]=listSplitIndices->at(i*4);
				m_pSplitIndices[i*4+1]=listSplitIndices->at(i*4+1)*4;
				m_pSplitIndices[i*4+2]=listSplitIndices->at(i*4+2)*4;
				m_pSplitIndices[i*4+3]=listSplitIndices->at(i*4+3)*4*UNIT_TRIANGLE_MAX;
			}

			int uIndex=0;
			for(int i=0;i<m_iNumSplitsUnitTri; i++)
			{
				for(int j=0; j<UNIT_TRIANGLE_MAX; j++)
				{
					if(j<listSplitUnitTriCount->at(i))
						m_pSplitUnitTriArray[uIndex]=listSplitUnitTriArray->at(i)[j];
					uIndex++;
				}
			}

			delete listSplitUnitTriArray;
			delete listSplitUnitTriCount;
			delete listSplitPlanes;
			delete listSplitIndices;
			delete mapIndexNode;
			delete mapNodeIndex;
		}
	}

	D3DXMATRIX*	BSPMesh::getSplitUnitTriArray()
	{
		return m_pSplitUnitTriArray;
	}
	D3DXPLANE*	BSPMesh::getSplitPlanes()
	{
		return m_pSplitPlanes;
	}
	int*		BSPMesh::getSplitIndices()
	{
		return m_pSplitIndices;
	}
	int 		BSPMesh::getNumSplits()
	{
		return m_iNumSplits;
	}
	int 		BSPMesh::getNumSplitsUnitTri()
	{
		return m_iNumSplitsUnitTri;
	}

	double BSPMesh::iterativeIntersect(Ray a_Ray, double t_in, double t_out)
	{
		BSPMesh* splitList[1024];
		double inList[1024];
		double outList[1024];

		unsigned int index=0;
		splitList[index]=this;
		inList[index]=t_in;
		outList[index]=t_out;
		index=1;

		while(index)
		{	
			index--;
			BSPMesh* current=splitList[index];
			t_in=inList[index];
			t_out=outList[index];

			if(!current->m_pLeft && !current->m_pRight)
				//TODO: enable bounding sphere build
				if(current->intersectBound(a_Ray)>=0.0)
				{
					double dist = current->intersectUnitTriangles(a_Ray);
					//double dist = current->intersectTriangles(a_Ray);
					if(dist<=t_out)
						return dist;
				}

				double orig=D3DXPlaneDot(&current->m_pStrategy->getSplitPlane(), &a_Ray.orig); 
				double dir=D3DXPlaneDot(&current->m_pStrategy->getSplitPlane(), &a_Ray.dir);
				double t_split=-orig/dir;
				double t_out_min=min(t_split, t_out);
				double t_in_max=max(t_split, t_in);

				BSPMesh* first;
				BSPMesh* second;
				if(dir>=0.0){
					first=current->m_pLeft;
					second=current->m_pRight;
				}
				else
				{
					first=current->m_pRight;
					second=current->m_pLeft;
				}

				if(second && t_split<=t_out)
				{
					splitList[index]=second;
					inList[index]=t_in_max;
					outList[index]=t_out;
					index++;
				}
				if(first && t_split>=t_in){
					splitList[index]=first;
					inList[index]=t_in;
					outList[index]=t_out_min;
					index++;
				}
		}
		return std::numeric_limits<double>::infinity();
	}

	double BSPMesh::rekursiveIntersect(Ray a_Ray, double t_in, double t_out)
	{
		if(!this->m_pLeft && !this->m_pRight)
			if(Mesh::intersectBound(a_Ray)>=0.0)
				return intersectUnitTriangles(a_Ray);

		double dist=std::numeric_limits<double>::infinity();
		double orig=D3DXPlaneDot(&m_pStrategy->getSplitPlane(), &a_Ray.orig); 
		double dir=D3DXPlaneDot(&m_pStrategy->getSplitPlane(), &a_Ray.dir);
		double t_split=-orig/dir;
		double t_out_min=min(t_split, t_out);
		double t_in_max=max(t_split, t_in);

		BSPMesh* first=this->m_pLeft;
		BSPMesh* second=this->m_pRight;

		//right was hit first if dir<0.0
		if(dir<0.0){
			first=this->m_pRight;
			second=this->m_pLeft;
		}

		if(first && t_split>=t_in){
			dist=first->rekursiveIntersect(a_Ray, t_in, t_out_min);
			if(dist<=t_out_min)	
				return dist;
		}
		if(second && t_split<=t_out)
			return second->rekursiveIntersect(a_Ray, t_in_max, t_out);

		return dist;
	}

	void BSPMesh::doSplit(void)
	{
		if(this->m_pStrategy && this->m_iIndices)
			this->splitMesh(this->m_pStrategy->getSplitPlane(), this, &this->m_pLeft, &this->m_pRight);

		this->m_pLeft->setStrategy(this->m_pStrategy->getLeftStrategy());
		this->m_pLeft->setPickableRoot(this->m_pRootPickable);
		if(this->m_pLeft->getMesh()->getNumVertices()==0)
		{
			delete this->m_pLeft;
			this->m_pLeft=NULL;
		}
		this->m_pRight->setStrategy(this->m_pStrategy->getRightStrategy());
		this->m_pRight->setPickableRoot(this->m_pRootPickable);
		if(this->m_pRight->getMesh()->getNumVertices()==0)
		{
			delete this->m_pRight;
			this->m_pRight=NULL;
		}
	}

	void BSPMesh::rekursiveSplit(void)
	{
		generateBoundingSphere();
		if(!this->m_pStrategy)
			return;
		if(this->m_iIndices==0)
			return;
		if(this->m_iIndices<=UNIT_TRIANGLE_MAX*3)
			return;
		if(this->m_pStrategy->getBisectionDepth()>BSP_MAX_DEPTH)
			return;

		doSplit();

		if(this->m_pLeft)
			this->m_pLeft->rekursiveSplit();

		if(this->m_pRight)
			this->m_pRight->rekursiveSplit();

		if(this->m_pLeft||this->m_pRight)
		{
			//this->m_iIndices=0;
			//delete[] this->m_pIndices;
			//this->m_pIndices=NULL;

			//this->m_iVertices=0;
			//delete[] this->m_pVertices;
			//this->m_pVertices=NULL;

			//this->m_iNumUnitTriangles=0;
			//if(this->m_pUnitTriangles)
			//	delete[] this->m_pUnitTriangles;
			//this->m_pUnitTriangles=NULL;
		}
	}

	////Use this method for Voxel rendering
	//void BSPMesh::rekursiveSplit(void)
	//{
	//	generateBoundingSphere();
	//	if(!this->m_pStrategy)
	//		return;
	//	if(this->m_iIndices==0)
	//			return;
	//	//if(this->m_pStrategy->getBisectionDepth()>26){
	//	if(this->m_pStrategy->getBisectionDepth()>24){
	//	//if(this->m_pStrategy->getBisectionDepth()>BSP_MAX_DEPTH)
	//			return;
	//	}
	//	doSplit();
	//
	//	if(this->m_pLeft)
	//		this->m_pLeft->rekursiveSplit();
	//
	//	if(this->m_pRight)
	//		this->m_pRight->rekursiveSplit();
	//	
	//	if(this->m_pLeft||this->m_pRight)
	//	{
	//		this->m_iIndices=0;
	//		delete[] this->m_pIndices;
	//		this->m_pIndices=NULL;
	//
	//		this->m_iVertices=0;
	//		delete[] this->m_pVertices;
	//		this->m_pVertices=NULL;
	//
	//		this->m_iNumUnitTriangles=0;
	//		if(this->m_pUnitTriangles)
	//			delete[] this->m_pUnitTriangles;
	//		this->m_pUnitTriangles=NULL;
	//	}
	//}

	void BSPMesh::generateBoundingSphere(void)
	{
		Sphere* pSphere=NULL;
		D3DXPLANE* pPlanes;
		unsigned int planeCount=0;
		pPlanes=this->m_pStrategy->getBoundingPlanes(&planeCount);
		BisectionCutMesh* left=NULL;
		BisectionCutMesh* right=NULL;
		BisectionCutMesh* leftTemp=NULL;
		//TODO: enable bounding sphere build
		for(unsigned int i=0; i<planeCount; i++)
		{
			if(!leftTemp)	BisectionCutMesh::cutMesh(pPlanes[i], this, &left, &right);
			else			BisectionCutMesh::cutMesh(pPlanes[i], leftTemp, &left, &right);
			delete right;
			if(leftTemp)	delete leftTemp;	
			leftTemp=left;
		}
		if(left)
		{
			Sphere s=left->getBoundingSphere();
			pSphere=new Sphere(s.center, s.radius);
			delete left;
		}

		if(pSphere){
			if(this->m_pBoundingSphere)
				delete this->m_pBoundingSphere;
			this->m_pBoundingSphere=pSphere;
		}
		else
			Mesh::generateBoundingSphere();
	}

	Mesh* BSPMesh::getMesh()
	{
		return this;
	}

	BSPMesh* BSPMesh::getLeftMesh()
	{
		return m_pLeft;
	}

	BSPMesh* BSPMesh::getRightMesh()
	{
		return m_pRight;
	}


	void BSPMesh::splitFace(const D3DXPLANE &plane, const Face& tri, std::vector<Face>* left, std::vector<Face>* right)
	{
		FLOAT distV0=D3DXPlaneDot(&plane, &tri.v0.pos);
		FLOAT distV1=D3DXPlaneDot(&plane, &tri.v1.pos);
		FLOAT distV2=D3DXPlaneDot(&plane, &tri.v2.pos);

		const char vertex0Left	=1;
		const char vertex0Right	=2;
		const char vertex0Middle=3;
		const char vertex1Left	=4;
		const char vertex1Right	=8;
		const char vertex1Middle=12;
		const char vertex2Left	=16;
		const char vertex2Right	=32;
		const char vertex2Middle=48;

		char triangleSide =0;
		if(distV0<0.0)		
			triangleSide	|=	vertex0Left;
		else if(distV0>0.0)	
			triangleSide	|=	vertex0Right;
		else				
			triangleSide	|=	vertex0Middle;

		if(distV1<0.0)		
			triangleSide	|=	vertex1Left;
		else if(distV1>0.0)	
			triangleSide	|=	vertex1Right;
		else				
			triangleSide	|=	vertex1Middle;

		if(distV2<0.0)		
			triangleSide	|=	vertex2Left;
		else if(distV2>0.0)	
			triangleSide	|=	vertex2Right;
		else				
			triangleSide	|=	vertex2Middle;

		switch(triangleSide)
		{
		case vertex0Left	| vertex1Left	| vertex2Left:
		case vertex0Left	| vertex1Left	| vertex2Middle:
		case vertex0Left	| vertex1Middle	| vertex2Left:
		case vertex0Middle	| vertex1Left	| vertex2Left:
		case vertex0Left	| vertex1Middle	| vertex2Middle:
		case vertex0Middle	| vertex1Left	| vertex2Middle:
		case vertex0Middle	| vertex1Middle	| vertex2Left:
			{
				left->push_back(tri);
			}
			break;
		case vertex0Right	| vertex1Right	| vertex2Right:
		case vertex0Right	| vertex1Right	| vertex2Middle:
		case vertex0Right	| vertex1Middle	| vertex2Right:
		case vertex0Middle	| vertex1Right	| vertex2Right:
		case vertex0Right	| vertex1Middle	| vertex2Middle:
		case vertex0Middle	| vertex1Right	| vertex2Middle:
		case vertex0Middle	| vertex1Middle	| vertex2Right:
			{
				right->push_back(tri);
			}
			break;
		default:
			{
				left->push_back(tri);
				right->push_back(tri);
			}
		}
	}

	void BSPMesh::splitMesh(const D3DXPLANE &plane, Mesh* pMesh, BSPMesh** ppLeftMesh, BSPMesh** ppRightMesh)
	{
		std::vector<Face>* left=new std::vector<Face>();
		std::vector<Face>* right=new std::vector<Face>();

		unsigned int faceIndices=pMesh->getNumIndices();
		Vertex* vertices=pMesh->getVertices();
		DWORD* indices=pMesh->getIndices();
		for(unsigned int i=0; i<faceIndices; i+=3)
		{
			Face tri;
			tri.v0=vertices[indices[i]];
			tri.v1=vertices[indices[i+1]];
			tri.v2=vertices[indices[i+2]];
			splitFace(plane, tri, left, right);
		}

		/////////////////////////////////////////////////
		unsigned int leftFaceIndices=(unsigned int)left->size()*3;
		Vertex* leftVertices=new Vertex[leftFaceIndices];
		DWORD* leftIndices=new DWORD[leftFaceIndices];
		for(unsigned int i=0; i<leftFaceIndices; i+=3)
		{
			leftVertices[i]=left->at(i/3).v0;
			leftVertices[i+1]=left->at(i/3).v1;
			leftVertices[i+2]=left->at(i/3).v2;
			leftIndices[i]=i;
			leftIndices[i+1]=i+1;
			leftIndices[i+2]=i+2;
		}
		*ppLeftMesh=new BSPMesh(leftIndices, leftVertices, leftFaceIndices, leftFaceIndices);
		delete[] leftVertices;
		delete[] leftIndices;

		////////////////////////////////////////////////
		unsigned int rightFaceIndices=(unsigned int)right->size()*3;
		Vertex* rightVertices=new Vertex[rightFaceIndices];
		DWORD* rightIndices=new DWORD[rightFaceIndices];
		for(unsigned int i=0; i<rightFaceIndices; i+=3)
		{
			rightVertices[i]=right->at(i/3).v0;
			rightVertices[i+1]=right->at(i/3).v1;
			rightVertices[i+2]=right->at(i/3).v2;
			rightIndices[i]=i;
			rightIndices[i+1]=i+1;
			rightIndices[i+2]=i+2;
		}
		*ppRightMesh=new BSPMesh(rightIndices, rightVertices, rightFaceIndices, rightFaceIndices);
		delete[] rightVertices;
		delete[] rightIndices;

		////////////////////////////////////////////
		delete left;
		delete right;
	}

}