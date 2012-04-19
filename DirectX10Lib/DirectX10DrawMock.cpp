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




#include "DirectX10DrawMock.h"
#include "DirectX10Renderer.h"
#include "DirectX10Effect.h"
namespace TheWhiteAmbit {

	DirectX10DrawMock::DirectX10DrawMock(DirectX10Renderer* a_pRenderer)
	{
		Node::Node();
		//////////////////////////////
		m_pRenderer = a_pRenderer;
		m_pEffect = NULL;
		m_pMesh = NULL;
		m_pSdkMesh = NULL;
		m_pObjMesh = NULL;
		m_pTechnique = NULL;
		m_pVertexLayout = NULL;
		m_pVertexBuffer = NULL;

		m_pWorldVariable = NULL;
		m_pViewVariable = NULL;
		m_pProjectionVariable = NULL;

		D3DXMatrixIdentity(&m_TransformMatrixWorld);
		D3DXMatrixIdentity(&m_TransformMatrixView);
		D3DXMatrixIdentity(&m_TransformMatrixProjection);

		HRESULT hr = S_OK;

		//// Create vertex buffer
		this->setPoints(
			D3DXVECTOR4( 0.0f, 0.5f, 0.5f, 1.0f ),
			D3DXVECTOR4( 0.5f,-0.5f, 0.5f, 1.0f ),
			D3DXVECTOR4(-0.5f,-0.5f, 0.5f, 1.0f ));
	}

	DirectX10DrawMock::~DirectX10DrawMock(void)
	{
		if( m_pVertexBuffer ) m_pVertexBuffer->Release();
		if( m_pVertexLayout ) m_pVertexLayout->Release();
	}

	D3DXMATRIX DirectX10DrawMock::transformEnterWorld(D3DXMATRIX a_pTransform)
	{
		m_TransformMatrixWorld=a_pTransform;
		return m_TransformMatrixWorld;
	}

	D3DXMATRIX DirectX10DrawMock::transformLeaveWorld(D3DXMATRIX a_pTransform)
	{
		return a_pTransform;
	}

	D3DXMATRIX DirectX10DrawMock::transformEnterView(D3DXMATRIX a_pTransform)
	{
		m_TransformMatrixView=a_pTransform;
		return m_TransformMatrixView;
	}

	D3DXMATRIX DirectX10DrawMock::transformLeaveView(D3DXMATRIX a_pTransform)
	{
		return a_pTransform;
	}

	D3DXMATRIX DirectX10DrawMock::transformEnterProjection(D3DXMATRIX a_pTransform)
	{
		m_TransformMatrixProjection=a_pTransform;
		return m_TransformMatrixProjection;
	}

	D3DXMATRIX DirectX10DrawMock::transformLeaveProjection(D3DXMATRIX a_pTransform)
	{
		return a_pTransform;
	}

	D3DXMATRIX DirectX10DrawMock::getWorld()
	{
		return m_TransformMatrixWorld;
	}
	D3DXMATRIX DirectX10DrawMock::getView()
	{
		return m_TransformMatrixView;
	}
	D3DXMATRIX DirectX10DrawMock::getProjection()
	{
		return m_TransformMatrixProjection;
	}
	Mesh* DirectX10DrawMock::getMesh()
	{
		if(!m_pMesh)
		{
			ID3DX10Mesh* pMesh=NULL;

			if(this->m_pObjMesh)
				pMesh = this->m_pObjMesh->getMesh();

			if(pMesh)
			{
				// Get the picked triangle
				DWORD* pIndices;
				Vertex* pVertices;
				//pVertices = (Vertex*)g_Mesh.GetRawVerticesAt(0);
				ID3DX10MeshBuffer*	pVertexBuffer;
				pMesh->GetVertexBuffer(0, &pVertexBuffer);
				SIZE_T	vertexBufferSize;
				pVertexBuffer->Map((void**)&pVertices, &vertexBufferSize);

				//pIndices = (DWORD*)g_Mesh.GetRawIndicesAt(0);
				ID3DX10MeshBuffer*	pIndicesBuffer;
				pMesh->GetIndexBuffer(&pIndicesBuffer);
				SIZE_T	indexBufferSize;
				pIndicesBuffer->Map((void**)&pIndices, &indexBufferSize);

				m_pMesh=new Mesh(pIndices, pVertices, pMesh->GetFaceCount() * 3, (UINT)vertexBufferSize / sizeof(Vertex) );

				pIndicesBuffer->Unmap();
				pVertexBuffer->Unmap();
			}
		}
		return m_pMesh;
	}

	double DirectX10DrawMock::intersect(Ray)
	{
		//TODO: implement picking for DirectX10DrawMock
		return 0.0;
	}

	void DirectX10DrawMock::setPoints(D3DXVECTOR4 a, D3DXVECTOR4 b, D3DXVECTOR4 c)
	{
		Vertex vertices[] =
		{
			a, D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( 0.0f, 0.5f ),
			b, D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( 0.5f, -0.5f ),
			c, D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( -0.5f, -0.5f ),
			a, D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( 0.0f, 0.5f ),
			b, D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( 0.5f, -0.5f ),
			c, D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( -0.5f, -0.5f ),
		};
		//for correct adjacency use acbacb so 0,2,4=a,b,c


		D3D10_BUFFER_DESC bd;
		bd.Usage = D3D10_USAGE_DEFAULT;
		bd.ByteWidth = sizeof( vertices );
		bd.BindFlags = D3D10_BIND_VERTEX_BUFFER;
		bd.CPUAccessFlags = 0;
		bd.MiscFlags = 0;
		D3D10_SUBRESOURCE_DATA InitData;
		InitData.pSysMem = &vertices;
		if(m_pVertexBuffer)
			m_pVertexBuffer->Release();
		HRESULT hr = m_pRenderer->getDevice()->CreateBuffer( &bd, &InitData, &m_pVertexBuffer );
	}

	void DirectX10DrawMock::setEffect(DirectX10Effect* a_pEffect)
	{
		HRESULT hr = S_OK;

		// Obtain the technique
		if(a_pEffect && a_pEffect->getEffect())
		{
			m_pWorldVariable = a_pEffect->getEffect()->GetVariableByName( "World" )->AsMatrix();
			m_pViewVariable = a_pEffect->getEffect()->GetVariableByName( "View" )->AsMatrix();
			m_pProjectionVariable = a_pEffect->getEffect()->GetVariableByName( "Projection" )->AsMatrix();

			// Define the input layout
			UINT numElements=0;
			D3D10_INPUT_ELEMENT_DESC* layout=Vertex::getVertexElements(&numElements);

			// Create the input layout
			m_pTechnique = a_pEffect->getEffect()->GetTechniqueByIndex(0);
			D3D10_PASS_DESC PassDesc;
			ZeroMemory( &PassDesc, sizeof( PassDesc ) );
			m_pTechnique->GetPassByIndex(0)->GetDesc( &PassDesc );

			hr = m_pRenderer->getDevice()->CreateInputLayout( layout, numElements, PassDesc.pIAInputSignature,
				PassDesc.IAInputSignatureSize, &m_pVertexLayout );
			if( FAILED( hr ) )
			{
				this->m_pEffect=NULL;
				return;
			}
			this->m_pEffect=a_pEffect;
		}
	}

	void DirectX10DrawMock::setEffectTime(double a_fTime)
	{
		//TODO: implement effect time for DX10 or move setter directly to Effect
		ID3D10EffectScalarVariable* pTime = m_pEffect->getEffect()->GetVariableByName( "g_fTime" )->AsScalar();
		pTime->SetFloat((FLOAT)a_fTime);
	}

	void DirectX10DrawMock::setMesh(DirectX10SdkMesh* a_pMesh)
	{
		this->m_pSdkMesh=a_pMesh;
	}

	void DirectX10DrawMock::setMesh(DirectX10ObjMesh* a_pMesh)
	{
		this->m_pObjMesh=a_pMesh;
	}

	void DirectX10DrawMock::render(void)
	{
		if(this->m_pEffect->getEffect())
		{
			m_pWorldVariable->SetMatrix( ( float* )&m_TransformMatrixWorld );
			m_pViewVariable->SetMatrix( ( float* )&m_TransformMatrixView );
			m_pProjectionVariable->SetMatrix( ( float* )&m_TransformMatrixProjection );

			D3DXMATRIX mWorldViewProjection=this->m_TransformMatrixWorld;
			D3DXMatrixMultiply(&mWorldViewProjection, &mWorldViewProjection, &this->m_TransformMatrixView);
			D3DXMatrixMultiply(&mWorldViewProjection, &mWorldViewProjection, &this->m_TransformMatrixProjection);

			ID3D10EffectMatrixVariable* mWorldViewProjectionVariable = m_pEffect->getEffect()->GetVariableByName( "WorldViewProjection" )->AsMatrix();
			mWorldViewProjectionVariable->SetMatrix(( float* )&mWorldViewProjection);
		}

		// Set the input layout
		if(m_pVertexLayout)
			m_pRenderer->getDevice()->IASetInputLayout( m_pVertexLayout );

		if(this->m_pEffect && !this->m_pSdkMesh && !this->m_pObjMesh)
		{
			// Set vertex buffer
			UINT stride = sizeof( Vertex );
			UINT offset = 0;
			m_pRenderer->getDevice()->IASetVertexBuffers( 0, 1, &m_pVertexBuffer, &stride, &offset );

			// Set primitive topology
			m_pRenderer->getDevice()->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ);

			if(m_pTechnique)
			{
				// Render a triangle
				D3D10_TECHNIQUE_DESC techDesc;

				m_pTechnique->GetDesc( &techDesc );
				for( UINT p = 0; p < techDesc.Passes; ++p )
				{
					m_pTechnique->GetPassByIndex( p )->Apply( 0 );
					m_pRenderer->getDevice()->Draw( 6, 0 );	//6 is number of vertices = 1 triangle with adjacency
				}
			}
		}
		else if(this->m_pEffect && this->m_pEffect->getEffect() &&this->m_pSdkMesh)
		{

			ID3D10EffectShaderResourceVariable* ptxDiffuseVariable = m_pEffect->getEffect()->GetVariableByName( "g_txDiffuse" )->AsShaderResource();
			ID3D10EffectShaderResourceVariable* ptxNormalVariable = m_pEffect->getEffect()->GetVariableByName( "g_txNormal" )->AsShaderResource();
			ID3D10EffectShaderResourceVariable* ptxSpecularVariable = m_pEffect->getEffect()->GetVariableByName( "g_txSpacular" )->AsShaderResource();

			this->m_pSdkMesh->getMesh()->Render( m_pRenderer->getDevice(), m_pTechnique, ptxDiffuseVariable, ptxNormalVariable, ptxSpecularVariable);
		}
		else if(this->m_pEffect && this->m_pEffect->getEffect() && this->m_pObjMesh)
		{	
			ID3D10EffectVectorVariable* m_pLightPosition = m_pEffect->getEffect()->GetVariableByName( "g_vLightPosition" )->AsVector();
			ID3D10EffectVectorVariable* m_pCameraPosition = m_pEffect->getEffect()->GetVariableByName( "g_vCameraPosition" )->AsVector();
			ID3D10EffectVectorVariable* m_pLightColor = m_pEffect->getEffect()->GetVariableByName( "g_vLightColor" )->AsVector();

			this->m_pObjMesh->Render(m_pEffect->getEffect());
		}
	}

	void DirectX10DrawMock::acceptEnter(TransformVisitor* a_pTransformVisitor)
	{
		a_pTransformVisitor->visitEnter(this);
		Node::acceptEnter(a_pTransformVisitor);
	}

	void DirectX10DrawMock::acceptLeave(TransformVisitor* a_pTransformVisitor)
	{
		a_pTransformVisitor->visitLeave(this);
		Node::acceptLeave(a_pTransformVisitor);
	}

	void DirectX10DrawMock::accept(RenderVisitor* a_pRenderVisitor)
	{
		a_pRenderVisitor->visit(this);
		Node::accept(a_pRenderVisitor);
	}

	void DirectX10DrawMock::accept(PickVisitor* a_pPickVisitor)
	{
		a_pPickVisitor->visit(this);
		Node::accept(a_pPickVisitor);
	}
}