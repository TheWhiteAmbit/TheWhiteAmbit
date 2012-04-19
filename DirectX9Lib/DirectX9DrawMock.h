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
#include <windows.h>
//#include <d3d9.h>
//#include <d3dx9.h>
#include "DirectX9Renderer.h"
#include "DirectX9Effect.h"
#include "DirectX9SdkMesh.h"
#include "DirectX9ObjMesh.h"
#include "DirectX9VertexBuffer.h"

#include "../SceneGraphLib/Node.h"
#include "../SceneGraphLib/BSPMesh.h"
#include "../SceneGraphLib/IRenderable.h"
#include "../SceneGraphLib/ITransformable.h"
#include "../SceneGraphLib/IPickable.h"
#include "../SceneGraphLib/IRaytraceable.h"
#include "../SceneGraphLib/TransformMemento.h"

#define D3DFVF_CUSTOMVERTEX 0
namespace TheWhiteAmbit {
	class DirectX9DrawMock :
		public Node, IRenderable, ITransformable, IRaytraceable, IOriginatorMementable
	{
		DirectX9VertexBuffer* m_pVertexBuffer; // Buffer to hold Vertices
		IDirect3DVertexDeclaration9* m_pVertexDecl;

		DirectX9SdkMesh*		m_pSdkMesh;
		DirectX9ObjMesh*		m_pObjMesh;
		DirectX9Renderer*		m_pRenderer;

		D3DXMATRIX				m_TransformMatrixWorld;
		D3DXMATRIX				m_TransformMatrixView;
		D3DXMATRIX				m_TransformMatrixProjection;

		//Remember matrices of previous rendered Frame
		D3DXMATRIX				m_LastTransformMatrixWorld;
		D3DXMATRIX				m_LastTransformMatrixView;
		D3DXMATRIX				m_LastTransformMatrixProjection;

		BSPMesh*				m_pBSPMesh;
		Sphere*					m_pBoundingSphere;

		//TransformMemento*		m_pLastTransformMemento;
		//int m_iDepth;
	public:
		void setPoints(D3DXVECTOR4 a, D3DXVECTOR4 b, D3DXVECTOR4 c);
		void setMesh(DirectX9SdkMesh* a_pMesh);
		void setMesh(DirectX9ObjMesh* a_pMesh);

		DirectX9DrawMock(DirectX9Renderer* a_pRenderer);
		virtual ~DirectX9DrawMock(void);

		void renderRek(BSPMesh* a_pMesh, int depth);

		//IRenderable
		virtual void render(IEffect*);
		//ITransformable
		virtual D3DXMATRIX transformEnterWorld(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformLeaveWorld(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformEnterView(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformLeaveView(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformEnterProjection(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformLeaveProjection(D3DXMATRIX a_pTransform);
		//IPickable
		virtual D3DXMATRIX getWorld();
		virtual D3DXMATRIX getView();
		virtual D3DXMATRIX getProjection();
		virtual D3DXMATRIX getLastWorld();
		virtual D3DXMATRIX getLastView();
		virtual D3DXMATRIX getLastProjection();
		virtual double intersect(Ray);
		//IRaytraceable
		virtual BSPMesh* getBSPMesh();
		virtual double intersectBound(Ray);
		//virtual Sphere getBoundingSphere();
		//virtual bool getBoundingTest(Ray);

		//IOriginatorMementable
		virtual IMementoMementable* saveToMemento(void);
		virtual void restoreFromMemento(IMementoMementable* memento);

		//Visitors
		virtual void acceptEnter(TransformVisitor* a_pTransformVisitor);
		virtual void acceptLeave(TransformVisitor* a_pTransformVisitor);
		virtual void accept(RenderVisitor* a_pRenderVisitor);
		virtual void accept(PickVisitor* a_pPickVisitor);
		virtual void accept(RaytraceVisitor* a_pPickVisitor);
		virtual void accept(MementoVisitor* a_pPickVisitor);
	};
}