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




#pragma comment (lib, "DirectXUTLib.lib")

#include "DirectX9SdkMesh.h"
namespace TheWhiteAmbit {
	DirectX9SdkMesh::DirectX9SdkMesh(DirectX9Renderer* a_pRenderer, LPCWSTR a_sFilename)
	{
		//if(FAILED( D3DXLoadMeshFromX(a_sFilename, NULL, a_pRenderer->getDevice(), &m_pAdnacency, &m_pMaterials, &m_pEffectInstances, &m_iNumMaterials, &m_pMesh)))
		//{
		//	m_pMesh=NULL;
		//	MessageBox( NULL,
		//                   L"The X file cannot be located.", L"Error", MB_OK );
		//       return;
		//}


		//D3DVERTEXELEMENT9 m_aVertDecl[] =
		//{
		//	{ 0, 0,  D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
		//	{ 0, 12, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL,   0 },
		//	{ 0, 24, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
		//	D3DDECL_END()
		//};
		//LPDIRECT3DVERTEXDECLARATION9    m_pVertDecl = NULL;
		//a_pRenderer->getDevice()->CreateVertexDeclaration( m_aVertDecl, &m_pVertDecl );
		//m_pMesh->SetVertexDecl( a_pRenderer->getDevice(), m_aVertDecl );

		m_pMesh=new CDXUTSDKMesh();
		m_pMesh->Create(a_pRenderer->getDevice(), a_sFilename, true);
	}

	DirectX9SdkMesh::~DirectX9SdkMesh(void)
	{
		if(m_pMesh)
		{
			//m_pMesh->Destroy();
			delete m_pMesh;
		}
	}

	CDXUTSDKMesh* DirectX9SdkMesh::getMesh(void)
	{
		return m_pMesh;
	}

	//// Render the objects
	//        for( int obj = 0; obj < NUM_OBJ; ++obj )
	//        {
	//            D3DXMATRIXA16 mWorldView = m_Obj[obj].m_mWorld;
	//            D3DXMatrixMultiply( &mWorldView, &mWorldView, pmView );
	//            V( m_pEffect->SetMatrix( "g_mWorldView", &mWorldView ) );
	//
	//            LPD3DXMESH pMesh = m_Obj[obj].m_Mesh.GetMesh();
	//            UINT cPass;
	//            V( m_pEffect->Begin( &cPass, 0 ) );
	//            for( UINT p = 0; p < cPass; ++p )
	//            {
	//                V( m_pEffect->BeginPass( p ) );
	//
	//                for( DWORD i = 0; i < m_Obj[obj].m_Mesh.m_dwNumMaterials; ++i )
	//                {
	//                    D3DXVECTOR4 vDif( m_Obj[obj].m_Mesh.m_pMaterials[i].Diffuse.r,
	//                                      m_Obj[obj].m_Mesh.m_pMaterials[i].Diffuse.g,
	//                                      m_Obj[obj].m_Mesh.m_pMaterials[i].Diffuse.b,
	//                                      m_Obj[obj].m_Mesh.m_pMaterials[i].Diffuse.a );
	//                    V( m_pEffect->SetVector( "g_vMaterial", &vDif ) );
	//                    if( m_Obj[obj].m_Mesh.m_pTextures[i] )
	//                        V( m_pEffect->SetTexture( "g_txScene", m_Obj[obj].m_Mesh.m_pTextures[i] ) )
	//                    else
	//                        V( m_pEffect->SetTexture( "g_txScene", m_pTexDef ) )
	//                    V( m_pEffect->CommitChanges() );
	//                    V( pMesh->DrawSubset( i ) );
	//                }
	//                V( m_pEffect->EndPass() );
	//            }
	//            V( m_pEffect->End() );
	//        }
}