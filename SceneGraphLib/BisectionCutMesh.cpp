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




#include "BisectionCutMesh.h"
#include "PickVisitor.h"
#include "Matrix.h"
#include <limits>

namespace TheWhiteAmbit {

	BisectionCutMesh::BisectionCutMesh(DWORD* a_pIndices, Vertex* a_pVertices, UINT a_iIndices, UINT a_iVertices)
		: Mesh(a_pIndices, a_pVertices, a_iIndices, a_iVertices)
	{
		m_pLeft=NULL;
		m_pRight=NULL;
		m_pStrategy=NULL;
		m_pRootPickable=NULL;
	}

	BisectionCutMesh::~BisectionCutMesh(void)
	{
	}

	void BisectionCutMesh::setStrategy(IBisectionStrategy* a_pStrategy)
	{
		this->m_pStrategy=a_pStrategy;
	}

	void BisectionCutMesh::setPickableRoot(IPickable* a_pPickable)
	{
		m_pRootPickable=a_pPickable;
	}

	void BisectionCutMesh::doSplit(void)
	{
		if(this->m_pStrategy && this->m_iIndices)
			this->cutMesh(this->m_pStrategy->getSplitPlane(), this, &this->m_pLeft, &this->m_pRight);

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

	void BisectionCutMesh::rekursiveSplit(void)
	{
		//generateUnitTriangles();
		if(!this->m_pStrategy)
			return;
		if(this->m_iIndices<=128*3)
			return;
		if(this->m_pStrategy->getBisectionDepth()>=32)
			return;
		doSplit();
		if(this->m_pLeft)
			this->m_pLeft->rekursiveSplit();
		if(this->m_pRight)
			this->m_pRight->rekursiveSplit();

		if(this->m_pLeft||this->m_pRight)
		{
			//this->getBoundingSphere();
			//this->m_iIndices=0;
			//this->m_iVertices=0;
			//delete[] this->m_pIndices;
			//delete[] this->m_pVertices;
			//this->m_pIndices=NULL;
			//this->m_pVertices=NULL;
			//if(this->m_pUnitTriangles)
			//	delete[] this->m_pUnitTriangles;
			//this->m_pUnitTriangles=NULL;
		}
	}

	Mesh* BisectionCutMesh::getMesh()
	{
		return this;
	}

	BisectionCutMesh* BisectionCutMesh::getLeftMesh()
	{
		return m_pLeft;
	}

	BisectionCutMesh* BisectionCutMesh::getRightMesh()
	{
		return m_pRight;
	}

	bool BisectionCutMesh::cutFace1and2(const D3DXPLANE &plane, const Face& tri,
		Face* t_out)
	{
		//calculate plane intersection points
		D3DXVECTOR4 p01;
		if(!D3DXPlaneIntersectLine((D3DXVECTOR3*)&p01, &plane, (D3DXVECTOR3*)&tri.v0.pos, (D3DXVECTOR3*)&tri.v1.pos))
			return false;
		D3DXVECTOR4 p02;
		if(!D3DXPlaneIntersectLine((D3DXVECTOR3*)&p02, &plane, (D3DXVECTOR3*)&tri.v0.pos, (D3DXVECTOR3*)&tri.v2.pos))
			return false;

		p01.w=1.0;
		p02.w=1.0;

		//calculate texture interpolation
		D3DXVECTOR2 t01;
		FLOAT edgeFull01	=	D3DXVec3Length((D3DXVECTOR3*)&(tri.v1.pos-tri.v0.pos));
		FLOAT edgeSplit01	=	D3DXVec3Length((D3DXVECTOR3*)&(p01-tri.v0.pos));
		D3DXVec2Lerp(&t01, &tri.v0.tex, &tri.v1.tex, edgeSplit01/edgeFull01);

		D3DXVECTOR2 t02;
		FLOAT edgeFull02	=	D3DXVec3Length((D3DXVECTOR3*)&(tri.v2.pos-tri.v0.pos));
		FLOAT edgeSplit02	=	D3DXVec3Length((D3DXVECTOR3*)&(p02-tri.v0.pos));
		D3DXVec2Lerp(&t02, &tri.v0.tex, &tri.v2.tex, edgeSplit02/edgeFull02);

		//calculate normal interpolation
		D3DXVECTOR4 n01;
		D3DXVec3Lerp((D3DXVECTOR3*)&n01, (D3DXVECTOR3*)&tri.v0.norm, (D3DXVECTOR3*)&tri.v1.norm, edgeSplit01/edgeFull01);
		D3DXVec3Normalize((D3DXVECTOR3*)&n01, (D3DXVECTOR3*)&n01);

		D3DXVECTOR4 n02;
		D3DXVec3Lerp((D3DXVECTOR3*)&n02, (D3DXVECTOR3*)&tri.v0.norm, (D3DXVECTOR3*)&tri.v2.norm, edgeSplit02/edgeFull02);
		D3DXVec3Normalize((D3DXVECTOR3*)&n02, (D3DXVECTOR3*)&n02);

		n01.w=0.0;
		n02.w=0.0;

		//generate first tri on side with one triangle
		t_out[0].v0.pos=tri.v0.pos;
		t_out[0].v1.pos=p01;
		t_out[0].v2.pos=p02;

		t_out[0].v0.tex=tri.v0.tex;
		t_out[0].v1.tex=t01;
		t_out[0].v2.tex=t02;

		t_out[0].v0.norm=tri.v0.norm;
		t_out[0].v1.norm=n01;
		t_out[0].v2.norm=n02;

		if(D3DXVec4LengthSq(&(tri.v2.pos-p01))<D3DXVec4LengthSq(&(tri.v1.pos-p02)))
		{
			//make two new triangles on side with quad
			//p01,v2,p02 and p01,v1,v2
			//use this to keep culling correctness
			t_out[1].v0.pos=p01;
			t_out[1].v1.pos=tri.v2.pos;
			t_out[1].v2.pos=p02;

			t_out[1].v0.tex=t01;
			t_out[1].v1.tex=tri.v2.tex;
			t_out[1].v2.tex=t02;

			t_out[1].v0.norm=n01;
			t_out[1].v1.norm=tri.v2.norm;
			t_out[1].v2.norm=n02;

			///////////////////////////////
			t_out[2].v0.pos=p01;
			t_out[2].v1.pos=tri.v1.pos;
			t_out[2].v2.pos=tri.v2.pos;

			t_out[2].v0.tex=t01;
			t_out[2].v1.tex=tri.v1.tex;
			t_out[2].v2.tex=tri.v2.tex;

			t_out[2].v0.norm=n01;
			t_out[2].v1.norm=tri.v1.norm;
			t_out[2].v2.norm=tri.v2.norm;	
		}
		else
		{
			//make two new triangles on side with quad
			//p01,v1,p02 and p02,v1,v2
			//use this to keep culling correctness
			t_out[1].v0.pos=p01;
			t_out[1].v1.pos=tri.v1.pos;
			t_out[1].v2.pos=p02;

			t_out[1].v0.tex=t01;
			t_out[1].v1.tex=tri.v1.tex;
			t_out[1].v2.tex=t02;

			t_out[1].v0.norm=n01;
			t_out[1].v1.norm=tri.v1.norm;
			t_out[1].v2.norm=n02;

			///////////////////////////////
			t_out[2].v0.pos=p02;
			t_out[2].v1.pos=tri.v1.pos;
			t_out[2].v2.pos=tri.v2.pos;

			t_out[2].v0.tex=t02;
			t_out[2].v1.tex=tri.v1.tex;
			t_out[2].v2.tex=tri.v2.tex;

			t_out[2].v0.norm=n02;
			t_out[2].v1.norm=tri.v1.norm;
			t_out[2].v2.norm=tri.v2.norm;
		}
		return true;
	}

	bool BisectionCutMesh::cutFace1and1(const D3DXPLANE &plane, const Face& tri,
		Face* t_out)
	{
		//calculate plane intersection points

		D3DXVECTOR4 p02;
		if(!D3DXPlaneIntersectLine((D3DXVECTOR3*)&p02, &plane, (D3DXVECTOR3*)&tri.v0.pos, (D3DXVECTOR3*)&tri.v2.pos))
			return false;

		p02.w=1.0;

		//calculate texture interpolation
		D3DXVECTOR2 t02;
		FLOAT edgeFull02	=	D3DXVec3Length((D3DXVECTOR3*)&(tri.v2.pos-tri.v0.pos));
		FLOAT edgeSplit02	=	D3DXVec3Length((D3DXVECTOR3*)&(p02-tri.v0.pos));
		D3DXVec2Lerp(&t02, &tri.v0.tex, &tri.v2.tex, edgeSplit02/edgeFull02);

		//calculate normal interpolation
		D3DXVECTOR4 n02;
		D3DXVec3Lerp((D3DXVECTOR3*)&n02, (D3DXVECTOR3*)&tri.v0.norm, (D3DXVECTOR3*)&tri.v2.norm, edgeSplit02/edgeFull02);
		D3DXVec3Normalize((D3DXVECTOR3*)&n02, (D3DXVECTOR3*)&n02);
		n02.w=0.0;

		//generate first tri on one side
		t_out[0].v0.pos=tri.v0.pos;
		t_out[0].v1.pos=tri.v1.pos;
		t_out[0].v2.pos=p02;

		t_out[0].v0.tex=tri.v0.tex;
		t_out[0].v1.tex=tri.v1.tex;
		t_out[0].v2.tex=t02;

		t_out[0].v0.norm=tri.v0.norm;
		t_out[0].v1.norm=tri.v1.norm;
		t_out[0].v2.norm=n02;

		//generate second tri on other side
		t_out[1].v0.pos=tri.v1.pos;
		t_out[1].v1.pos=tri.v2.pos;
		t_out[1].v2.pos=p02;

		t_out[1].v0.tex=tri.v1.tex;
		t_out[1].v1.tex=tri.v2.tex;
		t_out[1].v2.tex=t02;

		t_out[1].v0.norm=tri.v1.norm;
		t_out[1].v1.norm=tri.v2.norm;
		t_out[1].v2.norm=n02;
		return true;
	}


	void BisectionCutMesh::cutFace(const D3DXPLANE &plane, const Face& tri, std::vector<Face>* left, std::vector<Face>* right)
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
		case vertex0Middle	| vertex1Middle	| vertex2Middle:
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
		case vertex0Left | vertex1Right | vertex2Right:
			{
				Face t[3];
				cutFace1and2(plane, tri, &t[0]);
				left->push_back(t[0]);
				right->push_back(t[1]);
				right->push_back(t[2]);
			}
			break;
		case vertex0Right | vertex1Left | vertex2Right:
			{
				Face s;
				s.v0=tri.v1;
				s.v1=tri.v2;
				s.v2=tri.v0;
				Face t[3];
				cutFace1and2(plane, s, &t[0]);
				left->push_back(t[0]);
				right->push_back(t[1]);
				right->push_back(t[2]);
			}
			break;
		case vertex0Right | vertex1Right | vertex2Left:
			{
				Face s;
				s.v0=tri.v2;
				s.v1=tri.v0;
				s.v2=tri.v1;
				Face t[3];
				cutFace1and2(plane, s, &t[0]);
				left->push_back(t[0]);
				right->push_back(t[1]);
				right->push_back(t[2]);
			}
			break;
		case vertex0Right | vertex1Left | vertex2Left:
			{
				Face s;
				s.v0=tri.v0;
				s.v1=tri.v1;
				s.v2=tri.v2;
				Face t[3];
				cutFace1and2(plane, tri, &t[0]);
				right->push_back(t[0]);
				left->push_back(t[1]);
				left->push_back(t[2]);
			}
			break;
		case vertex0Left | vertex1Right | vertex2Left:
			{
				Face s;
				s.v0=tri.v1;
				s.v1=tri.v2;
				s.v2=tri.v0;
				Face t[3];
				cutFace1and2(plane, s, &t[0]);
				right->push_back(t[0]);
				left->push_back(t[1]);
				left->push_back(t[2]);
			}
			break;
		case vertex0Left | vertex1Left | vertex2Right:
			{
				Face s;
				s.v0=tri.v2;
				s.v1=tri.v0;
				s.v2=tri.v1;
				Face t[3];
				cutFace1and2(plane, s, &t[0]);
				right->push_back(t[0]);
				left->push_back(t[1]);
				left->push_back(t[2]);
			}
			break;
		case vertex0Left | vertex1Middle | vertex2Right:
			{
				Face t[2];
				cutFace1and1(plane, tri, &t[0]);
				left->push_back(t[0]);
				right->push_back(t[1]);
			}
			break;
		case vertex0Right | vertex1Middle | vertex2Left:
			{
				Face t[2];
				cutFace1and1(plane, tri, &t[0]);
				right->push_back(t[0]);
				left->push_back(t[1]);
			}
			break;
		case vertex0Middle | vertex1Left | vertex2Right:
			{
				Face s;
				s.v0=tri.v2;
				s.v1=tri.v0;
				s.v2=tri.v1;
				Face t[2];
				cutFace1and1(plane, s, &t[0]);
				right->push_back(t[0]);
				left->push_back(t[1]);
			}
			break;
		case vertex0Middle | vertex1Right | vertex2Left:
			{
				Face s;
				s.v0=tri.v2;
				s.v1=tri.v0;
				s.v2=tri.v1;
				Face t[2];
				cutFace1and1(plane, s, &t[0]);
				left->push_back(t[0]);
				right->push_back(t[1]);
			}
			break;
		case vertex0Left | vertex1Right | vertex2Middle:
			{
				Face s;
				s.v0=tri.v1;
				s.v1=tri.v2;
				s.v2=tri.v0;
				Face t[2];
				cutFace1and1(plane, s, &t[0]);
				right->push_back(t[0]);
				left->push_back(t[1]); 
			}
			break;
		case vertex0Right | vertex1Left | vertex2Middle:
			{
				Face s;
				s.v0=tri.v1;
				s.v1=tri.v2;
				s.v2=tri.v0;
				Face t[2];
				cutFace1and1(plane, s, &t[0]);
				left->push_back(t[0]);
				right->push_back(t[1]);
			}
			break;
		}
	}

	void BisectionCutMesh::cutMesh(const D3DXPLANE &plane, Mesh* pMesh, BisectionCutMesh** ppLeftMesh, BisectionCutMesh** ppRightMesh)
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
			cutFace(plane, tri, left, right);
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
		*ppLeftMesh=new BisectionCutMesh(leftIndices, leftVertices, leftFaceIndices, leftFaceIndices);
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
		*ppRightMesh=new BisectionCutMesh(rightIndices, rightVertices, rightFaceIndices, rightFaceIndices);
		delete[] rightVertices;
		delete[] rightIndices;

		////////////////////////////////////////////
		delete left;
		delete right;
	}

}