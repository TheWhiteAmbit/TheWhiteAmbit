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




#include "DirectX9ObjMesh.h"

namespace TheWhiteAmbit {

	DirectX9ObjMesh::DirectX9ObjMesh(DirectX9Renderer* a_pRenderer, LPCWSTR a_sFilename)
	{
		m_pEffect=NULL;
		// Create the mesh and load it with data already gathered from a file
		//HRESULT hr;
		if(FAILED(m_MeshLoader.Create( a_pRenderer->getDevice(), a_sFilename )))
		{
		}
	}

	ID3DXMesh* DirectX9ObjMesh::getMesh(void)
	{
		//TODO: implement for other meshes
		return this->m_MeshLoader.GetMesh();
	}

	void DirectX9ObjMesh::Render( ID3DXEffect* a_pEffect )
	{
		m_pEffect = a_pEffect;
		// Store the correct technique handles for each material
		for( UINT i = 0; i < m_MeshLoader.GetNumMaterials(); i++ )
		{
			Material* pMaterial = m_MeshLoader.GetMaterial( i );

			const char* strTechnique;

			if( pMaterial->pTextureDiffuse && pMaterial->bSpecular )
				strTechnique = "TexturedSpecular";
			else if( pMaterial->pTextureDiffuse && !pMaterial->bSpecular )
				strTechnique = "TexturedNoSpecular";
			else if( !pMaterial->pTextureDiffuse && pMaterial->bSpecular )
				strTechnique = "Specular";
			else if( !pMaterial->pTextureDiffuse && !pMaterial->bSpecular )
				strTechnique = "NoSpecular";

			pMaterial->hTechnique = m_pEffect->GetTechniqueByName( strTechnique );
		}

		UINT iCurSubset = -1;
		//UINT iCurSubset = ( UINT )( INT_PTR )m_SampleUI.GetComboBox( IDC_SUBSET )->GetSelectedData();

		// A subset of -1 was arbitrarily chosen to represent all subsets
		if( iCurSubset == -1 )
		{
			// Iterate through subsets, changing material properties for each
			for( UINT iSubset = 0; iSubset < m_MeshLoader.GetNumMaterials(); iSubset++ )
			{
				RenderSubset( iSubset );
			}
		}
		else
		{
			RenderSubset( iCurSubset );
		}
	}


	//--------------------------------------------------------------------------------------
	void DirectX9ObjMesh::RenderSubset( UINT iSubset )
	{
		HRESULT hr;
		UINT iPass, cPasses;

		// Retrieve the ID3DXMesh pointer and current material from the MeshLoader helper
		ID3DXMesh* pMesh = m_MeshLoader.GetMesh();
		Material* pMaterial = m_MeshLoader.GetMaterial( iSubset );

		D3DXHANDLE m_hAmbient = m_pEffect->GetParameterBySemantic( 0, "Ambient" );
		D3DXHANDLE m_hDiffuse = m_pEffect->GetParameterBySemantic( 0, "Diffuse" );
		D3DXHANDLE m_hSpecular = m_pEffect->GetParameterBySemantic( 0, "Specular" );
		D3DXHANDLE m_hTexture = m_pEffect->GetParameterBySemantic( 0, "Texture" );
		D3DXHANDLE m_hOpacity = m_pEffect->GetParameterBySemantic( 0, "Opacity" );
		D3DXHANDLE m_hSpecularPower = m_pEffect->GetParameterBySemantic( 0, "SpecularPower" );

		// Set the lighting variables and texture for the current material
		V( m_pEffect->SetValue( m_hAmbient, pMaterial->vAmbient, sizeof( D3DXVECTOR3 ) ) );
		V( m_pEffect->SetValue( m_hDiffuse, pMaterial->vDiffuse, sizeof( D3DXVECTOR3 ) ) );
		V( m_pEffect->SetValue( m_hSpecular, pMaterial->vSpecular, sizeof( D3DXVECTOR3 ) ) );
		//TODO: don't set texture form material of obj here...
		//V( m_pEffect->SetTexture( m_hTexture, pMaterial->pTextureDiffuse ) );
		V( m_pEffect->SetFloat( m_hOpacity, pMaterial->fAlpha ) );
		V( m_pEffect->SetInt( m_hSpecularPower, pMaterial->nShininess ) );

		V( m_pEffect->SetTechnique( pMaterial->hTechnique ) );
		V( m_pEffect->Begin( &cPasses, 0 ) );

		for( iPass = 0; iPass < cPasses; iPass++ )
		{
			V( m_pEffect->BeginPass( iPass ) );

			// The effect interface queues up the changes and performs them 
			// with the CommitChanges call. You do not need to call CommitChanges if 
			// you are not setting any parameters between the BeginPass and EndPass.
			// V( m_pEffect->CommitChanges() );

			// Render the mesh with the applied technique
			V( pMesh->DrawSubset( iSubset ) );

			V( m_pEffect->EndPass() );
		}
		V( m_pEffect->End() );
	}


	DirectX9ObjMesh::~DirectX9ObjMesh(void)
	{
		//m_MeshLoader.Destroy();
	}
}