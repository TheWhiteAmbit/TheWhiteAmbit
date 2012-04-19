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




#include "DirectX10TextureLayer.h"
namespace TheWhiteAmbit {

	DirectX10TextureLayer::DirectX10TextureLayer(DirectX10Renderer* a_pRenderer)
	{
		m_pRenderer=a_pRenderer;
		m_pEffect=NULL;
		m_pVertexLayout=NULL;
		m_pVertexBuffer=NULL;
		m_pTechnique=NULL;
		m_pTextureResource=NULL;
		m_pTexture=NULL;
		m_pTexture=new DirectX10Texture(m_pRenderer);

		m_ptxDiffuseVariable=NULL;

		fitVertexBufferTexcoords();
	}
	DirectX10TextureLayer::~DirectX10TextureLayer(void)
	{
		//TODO: implement this
	}

	void DirectX10TextureLayer::fitVertexBufferTexcoords(void)
	{
		FLOAT x_offset=0.0;
		FLOAT y_offset=0.0;

		if(this->m_pTexture)
		{
			x_offset=(FLOAT)(1.0/(this->m_pTexture->getWidth()*2));
			y_offset=(FLOAT)(1.0/(this->m_pTexture->getHeight()*2));
		}

		Vertex vertices[] =
		{
			D3DXVECTOR4( -1.0f,  1.0f, 0.5f, 1.0f),	D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( x_offset,		y_offset ),
			D3DXVECTOR4(  1.0f,  1.0f, 0.5f, 1.0f),	D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( 1.0f+x_offset,	y_offset ),
			D3DXVECTOR4( -1.0f, -1.0f, 0.5f, 1.0f),	D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( x_offset,		1.0f+y_offset ),
			D3DXVECTOR4(  1.0f, -1.0f, 0.5f, 1.0f),	D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( 1.0f+x_offset,	1.0f+y_offset ),
		};


		D3D10_BUFFER_DESC bd;
		bd.Usage = D3D10_USAGE_DEFAULT;
		bd.ByteWidth = sizeof( vertices );
		bd.BindFlags = D3D10_BIND_VERTEX_BUFFER;
		bd.CPUAccessFlags = 0;
		bd.MiscFlags = 0;

		D3D10_SUBRESOURCE_DATA InitData;
		InitData.pSysMem = &vertices;

		if(m_pVertexBuffer)
		{
			m_pVertexBuffer->Release();
			m_pVertexBuffer=NULL;
		}

		HRESULT hr=S_OK;
		if(FAILED(m_pRenderer->getDevice()->CreateBuffer( &bd, &InitData, &m_pVertexBuffer )))
		{
			m_pVertexBuffer=NULL;
			return;
		}
	}

	void DirectX10TextureLayer::setEffect(DirectX10Effect* a_pEffect)
	{
		// Obtain the technique
		if(a_pEffect && a_pEffect->getEffect())
		{
			//m_pWorldVariable = a_pEffect->getEffect()->GetVariableByName( "World" )->AsMatrix();
			//m_pViewVariable = a_pEffect->getEffect()->GetVariableByName( "View" )->AsMatrix();
			//m_pProjectionVariable = a_pEffect->getEffect()->GetVariableByName( "Projection" )->AsMatrix();

			m_ptxDiffuseVariable = a_pEffect->getEffect()->GetVariableByName("g_MeshTexture")->AsShaderResource(); 

			// Define the input layout
			UINT numElements=0;
			D3D10_INPUT_ELEMENT_DESC* layout=Vertex::getVertexElements(&numElements);

			// Create the input layout
			m_pTechnique = a_pEffect->getEffect()->GetTechniqueByIndex( 0 );
			D3D10_PASS_DESC PassDesc;
			ZeroMemory( &PassDesc, sizeof( PassDesc ) );
			m_pTechnique->GetPassByIndex(0)->GetDesc( &PassDesc );

			HRESULT hr=S_OK;
			if(FAILED(m_pRenderer->getDevice()->CreateInputLayout( layout, numElements, PassDesc.pIAInputSignature,
				PassDesc.IAInputSignatureSize, &m_pVertexLayout )))
			{  
				this->m_pEffect=NULL;
				return; 
			}
			this->m_pEffect=a_pEffect;
		}
	}
	void DirectX10TextureLayer::setTexture(DirectX10Texture* a_pTexture)
	{
		this->m_pTexture=a_pTexture;

		D3D10_SHADER_RESOURCE_VIEW_DESC desc;
		ZeroMemory(&desc, sizeof(desc));
		desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		//desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
		desc.Texture2D.MipLevels = 1;
		desc.Texture2D.MostDetailedMip = 0;
		desc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
		desc.Buffer.ElementOffset=0;
		desc.Buffer.ElementWidth=1;
		desc.Buffer.FirstElement=0;
		desc.Buffer.NumElements=1;

		//HRESULT hr=this->m_pRenderer->getDevice()->CreateShaderResourceView(this->m_pTexture->getTexture(), NULL, &m_pTextureResource);
		HRESULT hr=this->m_pRenderer->getDevice()->CreateShaderResourceView(this->m_pTexture->getTexture(), &desc, &m_pTextureResource);
		if(FAILED(hr))
			return;

		fitVertexBufferTexcoords();
	}
	//IRenderable
	void DirectX10TextureLayer::render(void)
	{
		if(		this->m_pEffect 
			&&	this->m_pVertexBuffer 
			&&	this->m_pVertexLayout 
			&&	this->m_pTexture 
			&&	this->m_pTextureResource
			&&	this->m_ptxDiffuseVariable)
		{ 
			//m_ptxDiffuseVariable->SetResource( m_pTextureResource );

			// Set vertex buffer
			UINT stride = sizeof( Vertex );
			UINT offset = 0;
			m_pRenderer->getDevice()->IASetVertexBuffers( 0, 1, &m_pVertexBuffer, &stride, &offset );

			// Set primitive topology
			m_pRenderer->getDevice()->IASetInputLayout( m_pVertexLayout );
			//m_pRenderer->getDevice()->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
			m_pRenderer->getDevice()->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_LINESTRIP);

			if(m_pTechnique)
			{
				// Render a triangle
				D3D10_TECHNIQUE_DESC techDesc;

				m_pTechnique->GetDesc( &techDesc );
				for( UINT p = 0; p < techDesc.Passes; ++p )
				{
					m_pTechnique->GetPassByIndex( p )->Apply( 0 );
					m_pRenderer->getDevice()->Draw( 4, 0 );	//4 is number of vertices = 1 quad
				}
			}
		}
	}
	//Visitors
	void DirectX10TextureLayer::accept(RenderVisitor* a_pRenderVisitor)
	{
		a_pRenderVisitor->visit(this);
		Node::accept(a_pRenderVisitor);
	}
}