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



#include "DirectX9DrawMock.h"
#include "../SceneGraphLib/Vertex.h"
#include "../SceneGraphLib/TetrahedronBisectionA0.h"
#include "../SceneGraphLib/HexahedronBisectionV.h"
#include <limits>
namespace TheWhiteAmbit {
	DirectX9DrawMock::DirectX9DrawMock(DirectX9Renderer* a_pRenderer)
	{
		Node::Node();
		//////////////////////////
		m_pBSPMesh = NULL;
		m_pBoundingSphere = NULL;

		m_pRenderer = a_pRenderer;
		m_pVertexBuffer = NULL;
		m_pVertexDecl=NULL;
		m_pSdkMesh = NULL;
		m_pObjMesh = NULL;	
		//m_iDepth=0;

		D3DXMatrixIdentity(&m_TransformMatrixWorld);
		D3DXMatrixIdentity(&m_TransformMatrixView);
		D3DXMatrixIdentity(&m_TransformMatrixProjection);


		HRESULT hr=S_OK;	
		if(m_pRenderer->getDevice())
			hr=m_pRenderer->getDevice()->CreateVertexDeclaration(Vertex::getVertexElements(), &m_pVertexDecl);
		if(FAILED(hr))
		{
			m_pVertexDecl=NULL;
			return;
		}

		//this->setPoints(
		//	D3DXVECTOR4( 0.0f, 0.5f, 0.5f, 1.0f ),
		//	D3DXVECTOR4( 0.5f,-0.5f, 0.5f, 1.0f ),
		//	D3DXVECTOR4(-0.5f,-0.5f, 0.5f, 1.0f ));
	}

	DirectX9DrawMock::~DirectX9DrawMock(void)
	{
		//TODO: make some resource management for VertexBuffer
		delete m_pVertexBuffer;
		if(m_pVertexDecl)
			m_pVertexDecl->Release();
	}

	void DirectX9DrawMock::setPoints(D3DXVECTOR4 a, D3DXVECTOR4 b, D3DXVECTOR4 c)
	{
		Vertex vertices[] =
		{
			a, D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( 0.0f, 0.5f ),
			b, D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( 0.5f, -0.5f ),
			c, D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( -0.5f, -0.5f ),
		};
		//TODO: make some resource management for VertexBuffer
		if(!m_pVertexBuffer)
			m_pVertexBuffer = new DirectX9VertexBuffer(m_pRenderer);
		this->m_pVertexBuffer->setVertices(&vertices[0], 3);
	}

	void DirectX9DrawMock::setMesh(DirectX9SdkMesh* a_pMesh)
	{
		this->m_pSdkMesh=a_pMesh;
	}

	void DirectX9DrawMock::setMesh(DirectX9ObjMesh* a_pMesh)
	{
		this->m_pObjMesh=a_pMesh;
	}

	//TODO: insert Effect code
	void DirectX9DrawMock::render(IEffect* a_pEffect)
	{
		DirectX9Effect* pEffect=(DirectX9Effect*)a_pEffect->getDirectX9Effect();
		if(pEffect)
			if(pEffect->getEffect())
			{
				pEffect->setValue(L"World", &this->m_TransformMatrixWorld);
				pEffect->setValue(L"View", &this->m_TransformMatrixView);
				pEffect->setValue(L"Projection", &this->m_TransformMatrixProjection);

				D3DXMATRIX worldViewProjection=this->m_TransformMatrixWorld;
				D3DXMatrixMultiply(&worldViewProjection, &worldViewProjection, &this->m_TransformMatrixView);
				D3DXMatrixMultiply(&worldViewProjection, &worldViewProjection, &this->m_TransformMatrixProjection);
				pEffect->setValue(L"WorldViewProjection", &worldViewProjection);

				//calculate transform matrices of last frame
				D3DXMATRIX lastWorldViewProjection=this->m_LastTransformMatrixWorld;
				D3DXMatrixMultiply(&lastWorldViewProjection, &lastWorldViewProjection, &this->m_LastTransformMatrixView);
				D3DXMatrixMultiply(&lastWorldViewProjection, &lastWorldViewProjection, &this->m_LastTransformMatrixProjection);			
				pEffect->setValue(L"LastWorld", &this->m_LastTransformMatrixWorld);
				pEffect->setValue(L"LastView", &this->m_LastTransformMatrixView);
				pEffect->setValue(L"LastProjection", &this->m_LastTransformMatrixProjection);
				pEffect->setValue(L"LastWorldViewProjection", &lastWorldViewProjection);
			}

			if(m_pVertexDecl)
				m_pRenderer->getDevice()->SetVertexDeclaration( m_pVertexDecl );
			if(m_pSdkMesh && pEffect && pEffect->getEffect()){

				D3DXHANDLE ptxDiffuseVariable;
				ptxDiffuseVariable= pEffect->getEffect()->GetParameterByName( NULL, "g_txDiffuse" );
				D3DXHANDLE ptxNormalVariable;
				ptxNormalVariable= pEffect->getEffect()->GetParameterByName( NULL, "g_txNormal" );
				D3DXHANDLE ptxSpecularVariable;
				ptxSpecularVariable= pEffect->getEffect()->GetParameterByName( NULL, "g_txSpecular" );

				this->m_pSdkMesh->getMesh()->Render( m_pRenderer->getDevice(), pEffect->getEffect(), pEffect->getEffect()->GetTechniqueByName("Render9"), ptxDiffuseVariable, ptxNormalVariable, ptxSpecularVariable);
				//this->m_pSdkMesh->getMesh()->Render( m_pRenderer->getDevice(), pEffect->getEffect(), pEffect->getEffect()->GetTechnique(0), ptxDiffuseVariable, ptxNormalVariable);
				//this->m_pSdkMesh->getMesh()->Render( m_pRenderer->getDevice(), pEffect->getEffect(), pEffect->getEffect()->GetTechniqueByName("Render9"), ptxDiffuseVariable);
				//this->m_pSdkMesh->getMesh()->Render( m_pRenderer->getDevice(), pEffect->getEffect(), pEffect->getEffect()->GetTechniqueByName("Render9"), 0, 0, 0);
			}
			else if(m_pObjMesh && pEffect && pEffect->getEffect()){
				// Cache the effect handles
				// TODO: set effect handles somewhere somehow somewhen
				//D3DXHANDLE m_hLightColor = pEffect->getEffect()->GetParameterBySemantic( 0, "LightColor" );
				//D3DXHANDLE m_hLightPosition = pEffect->getEffect()->GetParameterBySemantic( 0, "LightPosition" );
				//D3DXHANDLE m_hCameraPosition = pEffect->getEffect()->GetParameterBySemantic( 0, "CameraPosition" );

				this->m_pObjMesh->Render(pEffect->getEffect());
			}

			//TODO: remove since copying Matrices is done as Memento Pattern
			//Remember matrices of previous rendered Frame
			//m_LastTransformMatrixWorld=m_TransformMatrixWorld;
			//m_LastTransformMatrixView=m_TransformMatrixView;
			//m_LastTransformMatrixProjection=m_TransformMatrixProjection;
	}

	D3DXMATRIX DirectX9DrawMock::transformEnterWorld(D3DXMATRIX a_pTransform)
	{
		m_TransformMatrixWorld=a_pTransform;
		return m_TransformMatrixWorld;
	}

	D3DXMATRIX DirectX9DrawMock::transformLeaveWorld(D3DXMATRIX a_pTransform)
	{
		return a_pTransform;
	}

	D3DXMATRIX DirectX9DrawMock::transformEnterView(D3DXMATRIX a_pTransform)
	{
		m_TransformMatrixView=a_pTransform;
		return m_TransformMatrixView;
	}

	D3DXMATRIX DirectX9DrawMock::transformLeaveView(D3DXMATRIX a_pTransform)
	{
		return a_pTransform;
	}

	D3DXMATRIX DirectX9DrawMock::transformEnterProjection(D3DXMATRIX a_pTransform)
	{
		m_TransformMatrixProjection=a_pTransform;
		return m_TransformMatrixProjection;
	}

	D3DXMATRIX DirectX9DrawMock::transformLeaveProjection(D3DXMATRIX a_pTransform)
	{
		return a_pTransform;
	}

	D3DXMATRIX DirectX9DrawMock::getWorld()
	{
		return m_TransformMatrixWorld;
	}
	D3DXMATRIX DirectX9DrawMock::getView()
	{
		return m_TransformMatrixView;
	}
	D3DXMATRIX DirectX9DrawMock::getProjection()
	{
		return m_TransformMatrixProjection;
	}
	D3DXMATRIX DirectX9DrawMock::getLastWorld()
	{
		return m_LastTransformMatrixWorld;
	}
	D3DXMATRIX DirectX9DrawMock::getLastView()
	{
		return m_LastTransformMatrixView;
	}
	D3DXMATRIX DirectX9DrawMock::getLastProjection()
	{
		return m_LastTransformMatrixProjection;
	}
	BSPMesh* DirectX9DrawMock::getBSPMesh()
	{
		if(!m_pBSPMesh)
		{
			ID3DXMesh* pMesh=NULL;

			if(this->m_pSdkMesh)
				MessageBox(NULL, L"Not yet possible with SDKmesh", L"dont worry", 0);
			if(this->m_pObjMesh)
				pMesh=this->m_pObjMesh->getMesh();

			if(pMesh)
			{
				// Get the Picked triangle
				LPDIRECT3DVERTEXBUFFER9 pVB;
				LPDIRECT3DINDEXBUFFER9 pIB;

				pMesh->GetVertexBuffer( &pVB );
				pMesh->GetIndexBuffer( &pIB );

				DWORD* pIndices;
				Vertex* pVertices;

				// Determine the size of data to be moved into the vertex buffer.
				UINT nSizeOfIndexData = pMesh->GetNumFaces() * 3 * sizeof(DWORD);
				UINT nSizeOfVertexData = pMesh->GetNumVertices() * sizeof(Vertex);

				// Discard and refill the used portion of the vertex buffer.
				CONST DWORD dwLockFlags = D3DLOCK_DISCARD;
				//DWORD dwLockFlags = D3DLOCK_NOOVERWRITE;

				pIB->Lock( 0, nSizeOfIndexData, ( void** )&pIndices, dwLockFlags );
				pVB->Lock( 0, nSizeOfVertexData, ( void** )&pVertices, dwLockFlags );

				BSPMesh* bspMesh=new BSPMesh(pIndices, pVertices, pMesh->GetNumFaces() * 3, pMesh->GetNumVertices());

				pVB->Unlock();
				pIB->Unlock();

				Tetrahedron t(
					D3DXVECTOR4( 12, 8,-4, 1),
					D3DXVECTOR4(-4, 8,-4, 1),
					D3DXVECTOR4(-4,-8,-4, 1), 
					D3DXVECTOR4(-4,-8, 12, 1));
				Hexahedron h(
					D3DXVECTOR4(-2,-2,-2, 1),
					D3DXVECTOR4( 2,-2,-2, 1),
					D3DXVECTOR4(-2, 2,-2, 1),
					D3DXVECTOR4( 2, 2,-2, 1),
					D3DXVECTOR4(-2,-2, 2, 1),
					D3DXVECTOR4( 2,-2, 2, 1),
					D3DXVECTOR4(-2, 2, 2, 1),
					D3DXVECTOR4( 2, 2, 2, 1));

				//TODO: select different BSP-Strategys here
				//bspMesh->setStrategy(new TetrahedronBisectionA0(t, 1));
				bspMesh->setStrategy(new HexahedronBisectionV(h, 1));
				bspMesh->setPickableRoot(this);
				bspMesh->rekursiveSplit();

				m_pBSPMesh=bspMesh;
			}
		}
		return m_pBSPMesh;
	}

	double DirectX9DrawMock::intersect(Ray a_Ray)
	{
		//TODO: add pickable to bsp mesh
		return -std::numeric_limits<double>::infinity();
	}

	double DirectX9DrawMock::intersectBound(Ray a_Ray)
	{
		if(this->getBSPMesh())
		{
			return m_pBSPMesh->intersectBound(a_Ray);
		}
		return -std::numeric_limits<double>::infinity();
	}

	IMementoMementable* DirectX9DrawMock::saveToMemento(void){
		//TODO: this gives a TransformMemento
		TransformMemento* pTransformMemento=new TransformMemento();
		pTransformMemento->setWorld(this->m_TransformMatrixWorld);
		pTransformMemento->setView(this->m_TransformMatrixView);
		pTransformMemento->setProjection(this->m_TransformMatrixProjection);
		return (IMementoMementable*)pTransformMemento;
	}

	void DirectX9DrawMock::restoreFromMemento(IMementoMementable* memento){
		//TODO: restore function from memento - hack expects TransformMemento
		TransformMemento* pTransformMemento = (TransformMemento*)memento;
		D3DXMATRIX dummy;
		D3DXMatrixIdentity(&dummy);
		m_LastTransformMatrixWorld=pTransformMemento->transformEnterWorld(dummy);
		m_LastTransformMatrixView=pTransformMemento->transformEnterView(dummy);
		m_LastTransformMatrixProjection=pTransformMemento->transformEnterProjection(dummy);
	}


	void DirectX9DrawMock::acceptEnter(TransformVisitor* a_pTransformVisitor)
	{
		a_pTransformVisitor->visitEnter(this);
		Node::acceptEnter(a_pTransformVisitor);
	}

	void DirectX9DrawMock::acceptLeave(TransformVisitor* a_pTransformVisitor)
	{
		a_pTransformVisitor->visitLeave(this);
		Node::acceptLeave(a_pTransformVisitor);
	}

	void DirectX9DrawMock::accept(RenderVisitor* a_pRenderVisitor)
	{
		a_pRenderVisitor->visit(this);
		Node::accept(a_pRenderVisitor);
	}

	void DirectX9DrawMock::accept(PickVisitor* a_pPickVisitor)
	{
		a_pPickVisitor->visit(this);
		Node::accept(a_pPickVisitor);
	}

	void DirectX9DrawMock::accept(RaytraceVisitor* a_pVisitor)
	{
		a_pVisitor->visit(this);
		Node::accept(a_pVisitor);
	}

	void DirectX9DrawMock::accept(MementoVisitor* a_pVisitor)
	{
		a_pVisitor->visit(this);
		Node::accept(a_pVisitor);
	}
}