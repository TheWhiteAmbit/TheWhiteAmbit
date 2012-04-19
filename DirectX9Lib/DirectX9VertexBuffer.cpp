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




#include "DirectX9VertexBuffer.h"
namespace TheWhiteAmbit {
	DirectX9VertexBuffer::DirectX9VertexBuffer(DirectX9Renderer* a_pRenderer)
	{
		this->m_pRenderer=a_pRenderer;
		this->m_pVertexBuffer=NULL;
		this->m_iVertexCount=0;
	}

	DirectX9VertexBuffer::~DirectX9VertexBuffer(void)
	{
		if(this->m_pVertexBuffer)
			this->m_pVertexBuffer->Release();
	}

	void DirectX9VertexBuffer::setMesh(Mesh* a_pMesh)
	{
		if(a_pMesh->getNumVertices()!=m_iVertexCount)
		{
			if(this->m_pVertexBuffer)
				this->m_pVertexBuffer->Release();
			this->m_pVertexBuffer=NULL;
			this->m_iVertexCount=0;
			if( FAILED( m_pRenderer->getDevice()->CreateVertexBuffer( sizeof( Vertex )*a_pMesh->getNumVertices(),
				0, NULL,
				D3DPOOL_DEFAULT, &m_pVertexBuffer, NULL ) ) )
			{	
				this->m_pVertexBuffer=NULL;	
				return;	
			}
		}

		// Now we fill the vertex buffer. To do this, we need to Lock() the VB to
		// gain access to the Vertices. This mechanism is required becuase vertex
		// buffers may be in device memory.
		VOID* pVertices;
		if( SUCCEEDED( m_pVertexBuffer->Lock( 0, sizeof( Vertex )*a_pMesh->getNumVertices(), ( void** )&pVertices, 0 ) ) )
		{	
			memcpy( pVertices, a_pMesh->getVertices(), sizeof( Vertex )*a_pMesh->getNumVertices() );
			m_pVertexBuffer->Unlock();
			this->m_iVertexCount=a_pMesh->getNumVertices();
		}
	}

	void DirectX9VertexBuffer::setVertices(Vertex* a_pVertices, unsigned int a_iNumVertices)
	{
		if(a_iNumVertices!=m_iVertexCount)
		{
			if(this->m_pVertexBuffer)
				this->m_pVertexBuffer->Release();
			this->m_pVertexBuffer=NULL;
			this->m_iVertexCount=0;
			if( FAILED( m_pRenderer->getDevice()->CreateVertexBuffer( sizeof( Vertex )*a_iNumVertices,
				0, NULL,
				D3DPOOL_DEFAULT, &m_pVertexBuffer, NULL ) ) )
			{	
				this->m_pVertexBuffer=NULL;	
				return;	
			}
		}

		VOID* pVertices;
		if( SUCCEEDED( m_pVertexBuffer->Lock( 0, sizeof( Vertex )*a_iNumVertices, ( void** )&pVertices, 0 ) ) )
		{
			memcpy( pVertices, a_pVertices, sizeof( Vertex )*a_iNumVertices );
			m_pVertexBuffer->Unlock();
			this->m_iVertexCount=a_iNumVertices;
		}
	}

	IDirect3DVertexBuffer9* DirectX9VertexBuffer::getVertexBuffer()
	{
		return this->m_pVertexBuffer;
	}

	unsigned int DirectX9VertexBuffer::getNumVertices()
	{
		return this->m_iVertexCount;
	}

	void DirectX9VertexBuffer::Render(ID3DXEffect* a_pEffect)
	{
		if(a_pEffect){
			m_pRenderer->getDevice()->SetStreamSource( 0, m_pVertexBuffer, 0, sizeof( Vertex ) );

			UINT iPass, cPasses;
			a_pEffect->Begin( &cPasses, 0 );
			for( iPass = 0; iPass < cPasses; iPass++ )
			{
				a_pEffect->BeginPass( iPass );
				m_pRenderer->getDevice()->DrawPrimitive( D3DPT_TRIANGLELIST, 0, (m_iVertexCount/3) );
				a_pEffect->EndPass();
			}
			a_pEffect->End();
		}
	}
}