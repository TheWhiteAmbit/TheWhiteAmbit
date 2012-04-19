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




#include "DirectX9TextureLayer.h"
//#include "../SceneGraphLib/RenderVisitor.h"

#define D3DFVF_CUSTOMVERTEX 0
namespace TheWhiteAmbit {
	DirectX9TextureLayer::DirectX9TextureLayer(DirectX9Renderer* a_pRenderer)
	{
		upperLeft.x = -1.0f;
		upperLeft.y = -1.0f;
		upperLeft.z = 0.0f;
		upperLeft.w = 1.0f;
		upperRight.x = 1.0f;
		upperRight.y = -1.0f;
		upperRight.z = 0.0f;
		upperRight.w = 1.0f;
		lowerLeft.x = -1.0f;
		lowerLeft.y = 1.0f;
		lowerLeft.z = 0.0f;
		lowerLeft.w = 1.0f;
		lowerRight.x = 1.0f;
		lowerRight.y = 1.0f;
		lowerRight.z = 0.0f;
		lowerRight.w = 1.0f;

		m_pRenderer=a_pRenderer;
		m_pVertexDecl=NULL;
		m_pVertexBuffer=NULL;
		m_pTexture=NULL;
		//m_pTexture=new DirectX9Texture(this->m_pRenderer);

		//Create vertex Layout
		HRESULT hr=S_OK;
		if(m_pRenderer->getDevice())
			hr=m_pRenderer->getDevice()->CreateVertexDeclaration(Vertex::getVertexElements(), &m_pVertexDecl);
		if(FAILED(hr))
		{
			m_pVertexDecl=NULL;
			return;
		}

		// Create the vertex buffer. Here we are allocating enough memory
		// (from the default pool) to hold all our 3 custom Vertices. We don't
		// specify the FVF as we are using D3DVERTEXELEMENT9 layout.
		if( FAILED( m_pRenderer->getDevice()->CreateVertexBuffer( 4 * sizeof( Vertex ),
			0, D3DFVF_CUSTOMVERTEX,
			D3DPOOL_DEFAULT, &m_pVertexBuffer, NULL ) ) )
		{
			m_pVertexBuffer=NULL;
			return;
		}

		fitVertexBufferTexcoords();
	}

	DirectX9TextureLayer::~DirectX9TextureLayer(void)
	{
		if( m_pVertexBuffer != NULL )
			m_pVertexBuffer->Release();
		if( m_pVertexDecl != NULL )
			m_pVertexDecl->Release();
	}

	//TODO: insert Effect code
	void DirectX9TextureLayer::render(IEffect* a_pEffect)
	{
		DirectX9Effect* pEffect = (DirectX9Effect*) a_pEffect->getDirectX9Effect();
		if(pEffect && pEffect->getEffect() && m_pTexture)
		{
			D3DXHANDLE ptxDiffuseVariable;
			ptxDiffuseVariable= pEffect->getEffect()->GetParameterByName( NULL, "g_MeshTexture" );
			pEffect->getEffect()->SetTexture( ptxDiffuseVariable, this->m_pTexture->getTexture() );

			if(m_pVertexDecl)
				m_pRenderer->getDevice()->SetVertexDeclaration( m_pVertexDecl );
			m_pRenderer->getDevice()->SetStreamSource( 0, m_pVertexBuffer, 0, sizeof( Vertex ) );

			//TODO: set texture on effect
			UINT iPass, cPasses;
			pEffect->getEffect()->Begin( &cPasses, 0 );
			for( iPass = 0; iPass < cPasses; iPass++ )
			{
				pEffect->getEffect()->BeginPass( iPass );
				m_pRenderer->getDevice()->DrawPrimitive( D3DPT_TRIANGLESTRIP, 0, 2 );
				pEffect->getEffect()->EndPass();
			}
			pEffect->getEffect()->End();
		}
	}

	void DirectX9TextureLayer::fitVertexBufferTexcoords(void)
	{
		FLOAT x_offset=0.0;
		FLOAT y_offset=0.0;

		if(this->m_pTexture)
		{
			x_offset=(FLOAT)(.5/(this->m_pTexture->getWidth()));
			y_offset=(FLOAT)(.5/(this->m_pTexture->getHeight()));	
		}

		if(this->m_pVertexBuffer)
		{
			Vertex vertices[] =
			{
				lowerRight,	D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( 0.0f         ,	0.0f ),
				lowerLeft,	D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( 1.0f-x_offset,	0.0f ),
				upperRight,	D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( 0.0f         ,	1.0f -y_offset),
				upperLeft,	D3DXVECTOR4( 0.0f, 0.0f, 1.0f, 0.0f ), D3DXVECTOR2( 1.0f-x_offset,	1.0f -y_offset),
			};

			// Now we fill the vertex buffer. To do this, we need to Lock() the VB to
			// gain access to the Vertices. This mechanism is required becuase vertex
			// buffers may be in device memory.
			VOID* pVertices;
			if( FAILED( m_pVertexBuffer->Lock( 0, sizeof( vertices ), ( void** )&pVertices, 0 ) ) )
			{
				return;
			}
			memcpy( pVertices, vertices, sizeof( vertices ) );
			m_pVertexBuffer->Unlock();
		}
	}

	void DirectX9TextureLayer::setYOrthogonalPosition(double minX, double minY, double maxX, double maxY){
		minX=minX*2.0-1.0;
		minY=minY*2.0-1.0;
		maxX=maxX*2.0-1.0;
		maxY=maxY*2.0-1.0;

		upperLeft.x = (FLOAT)minX;
		upperLeft.y = 0.0;
		upperLeft.z = (FLOAT)minY;

		upperRight.x = (FLOAT)maxX;
		upperRight.y = 0.0;
		upperRight.z = (FLOAT)minY;

		lowerLeft.x = (FLOAT)minX;
		lowerLeft.y = 0.0;
		lowerLeft.z = (FLOAT)maxY;

		lowerRight.x = (FLOAT)maxX;
		lowerRight.y = 0.0;
		lowerRight.z = (FLOAT)maxY;

		fitVertexBufferTexcoords();
	}

	void DirectX9TextureLayer::setZOrthogonalPosition(double minX, double minY, double maxX, double maxY){
		minX=minX*2.0-1.0;
		minY=minY*2.0-1.0;
		maxX=maxX*2.0-1.0;
		maxY=maxY*2.0-1.0;

		upperLeft.x = (FLOAT)minX;
		upperLeft.y = (FLOAT)minY;
		upperLeft.z = 0.0;

		upperRight.x = (FLOAT)maxX;
		upperRight.y = (FLOAT)minY;
		upperRight.z = 0.0;

		lowerLeft.x = (FLOAT)minX;
		lowerLeft.y = (FLOAT)maxY;
		lowerLeft.z = 0.0;

		lowerRight.x = (FLOAT)maxX;
		lowerRight.y = (FLOAT)maxY;
		lowerRight.z = 0.0;

		fitVertexBufferTexcoords();
	}

	void DirectX9TextureLayer::setTextureSource(unsigned int a_iTextureNumber, DirectX9Texture* a_pTexture)
	{
		switch(a_iTextureNumber) {
		case 0:
			this->m_pTexture=a_pTexture;
			break;
		default:
			break;
		}
		fitVertexBufferTexcoords();
	}

	DirectX9Texture* DirectX9TextureLayer::getTextureSource(unsigned int a_iTextureNumber)
	{
		switch(a_iTextureNumber) {
		case 0:
			return this->m_pTexture;
			break;
		default:
			return NULL;
		}
	}

	void DirectX9TextureLayer::accept(RenderVisitor* a_pRenderVisitor)
	{
		a_pRenderVisitor->visit(this);
		Node::accept(a_pRenderVisitor);
	}
}