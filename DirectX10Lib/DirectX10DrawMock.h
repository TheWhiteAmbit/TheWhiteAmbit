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

#include "DirectX10Renderer.h"
#include "DirectX10Effect.h"
#include "DirectX10SdkMesh.h"
#include "DirectX10ObjMesh.h"

#include "../SceneGraphLib/Node.h"
#include "../SceneGraphLib/IRenderable.h"
#include "../SceneGraphLib/ITransformable.h"
#include "../SceneGraphLib/IPickable.h"

#include "../SceneGraphLib/RenderVisitor.h"
#include "../SceneGraphLib/TransformVisitor.h"
#include "../SceneGraphLib/PickVisitor.h"
namespace TheWhiteAmbit {
	class DirectX10DrawMock :
		public Node, IRenderable, ITransformable, IPickable
	{
		DirectX10Effect*        m_pEffect;
		DirectX10SdkMesh*		m_pSdkMesh;
		DirectX10ObjMesh*		m_pObjMesh;
		ID3D10EffectTechnique*  m_pTechnique;
		ID3D10InputLayout*      m_pVertexLayout;
		ID3D10Buffer*           m_pVertexBuffer;
		DirectX10Renderer*		m_pRenderer;

		ID3D10EffectMatrixVariable* m_pWorldVariable;
		ID3D10EffectMatrixVariable* m_pViewVariable;
		ID3D10EffectMatrixVariable* m_pProjectionVariable;

		D3DXMATRIX m_TransformMatrixWorld;
		D3DXMATRIX m_TransformMatrixView;
		D3DXMATRIX m_TransformMatrixProjection;

		Mesh*		m_pMesh;
	public:
		DirectX10DrawMock(DirectX10Renderer* a_pRenderer);
		virtual ~DirectX10DrawMock(void);


		//IRenderable
		virtual void render(void);
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
		virtual Mesh* getMesh();
		virtual double intersect(Ray);
		//Visitors
		virtual void acceptEnter(TransformVisitor* a_pTransformVisitor);
		virtual void acceptLeave(TransformVisitor* a_pTransformVisitor);
		virtual void accept(RenderVisitor* a_pRenderVisitor);	
		virtual void accept(PickVisitor* a_pPickVisitor);

		void setEffect(DirectX10Effect* a_pEffect);
		void setMesh(DirectX10SdkMesh* a_pMesh);
		void setMesh(DirectX10ObjMesh* a_pMesh);
		void setEffectTime(double a_fTime);
		void setPoints(D3DXVECTOR4 a, D3DXVECTOR4 b, D3DXVECTOR4 c);
		ID3D10Effect* getEffect(void);
	};
}