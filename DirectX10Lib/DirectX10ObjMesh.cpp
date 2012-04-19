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




#include "DirectX10ObjMesh.h"
#include "DirectX10Effect.h"
namespace TheWhiteAmbit {
	DirectX10ObjMesh::DirectX10ObjMesh(DirectX10Renderer* a_pRenderer, LPCWSTR a_sFilename)
	{
		m_pEffect=NULL;
		// Create the mesh and load it with data already gathered from a file
		//HRESULT hr;
		if(FAILED(m_MeshLoader.Create( a_pRenderer->getDevice(), a_sFilename )))
		{
		}
	}

	void DirectX10ObjMesh::Render( ID3D10Effect* a_pEffect )
	{
		m_pEffect=a_pEffect;
		// Store the correct technique for each material
		for ( UINT i = 0; i < m_MeshLoader.GetNumMaterials(); ++i )
		{
			Material* pMaterial = m_MeshLoader.GetMaterial( i );

			const char* strTechnique = "";

			if( pMaterial->pTextureRV10 && pMaterial->bSpecular )
				strTechnique = "TexturedSpecular";
			else if( pMaterial->pTextureRV10 && !pMaterial->bSpecular )
				strTechnique = "TexturedNoSpecular";
			else if( !pMaterial->pTextureRV10 && pMaterial->bSpecular )
				strTechnique = "Specular";
			else if( !pMaterial->pTextureRV10 && !pMaterial->bSpecular )
				strTechnique = "NoSpecular";

			pMaterial->pTechnique = m_pEffect->GetTechniqueByName( strTechnique );
		}



		UINT iCurSubset = -1;
		//UINT iCurSubset = ( UINT )( INT_PTR )g_SampleUI.GetComboBox( IDC_SUBSET )->GetSelectedData();

		//
		// Render the mesh
		//
		if ( iCurSubset == -1 )
		{
			for ( UINT iSubset = 0; iSubset < m_MeshLoader.GetNumSubsets(); ++iSubset )
			{
				RenderSubset( iSubset );
			}
		} else
		{
			RenderSubset( iCurSubset );
		}
	}

	void DirectX10ObjMesh::RenderSubset( UINT iSubset )
	{
		HRESULT hr;

		Material* pMaterial = m_MeshLoader.GetSubsetMaterial( iSubset );

		ID3D10EffectVectorVariable* m_pAmbient = m_pEffect->GetVariableByName( "g_vMaterialAmbient" )->AsVector();
		ID3D10EffectVectorVariable* m_pDiffuse = m_pEffect->GetVariableByName( "g_vMaterialDiffuse" )->AsVector();
		ID3D10EffectVectorVariable* m_pSpecular = m_pEffect->GetVariableByName( "g_vMaterialSpecular" )->AsVector();
		ID3D10EffectScalarVariable* m_pOpacity = m_pEffect->GetVariableByName( "g_fMaterialAlpha" )->AsScalar();
		ID3D10EffectScalarVariable* m_pSpecularPower = m_pEffect->GetVariableByName( "g_nMaterialShininess" )->AsScalar();

		V( m_pAmbient->SetFloatVector( pMaterial->vAmbient ) );
		V( m_pDiffuse->SetFloatVector( pMaterial->vDiffuse ) );
		V( m_pSpecular->SetFloatVector( pMaterial->vSpecular ) );
		V( m_pOpacity->SetFloat( pMaterial->fAlpha ) );
		V( m_pSpecularPower->SetInt( pMaterial->nShininess ) );


		if ( !IsErrorResource( pMaterial->pTextureRV10 ) )
		{
			ID3D10EffectShaderResourceVariable* m_ptxDiffuseVariable = m_pEffect->GetVariableByName( "g_MeshTexture" )->AsShaderResource();    
			m_ptxDiffuseVariable->SetResource( pMaterial->pTextureRV10 );
		}

		D3D10_TECHNIQUE_DESC techDesc;
		pMaterial->pTechnique->GetDesc( &techDesc );

		for ( UINT p = 0; p < techDesc.Passes; ++p )
		{
			pMaterial->pTechnique->GetPassByIndex(p)->Apply(0);
			m_MeshLoader.GetMesh()->DrawSubset(iSubset);
		}
	}

	DirectX10ObjMesh::~DirectX10ObjMesh(void)
	{
		m_MeshLoader.Destroy();
	}

	ID3DX10Mesh* DirectX10ObjMesh::getMesh(void)
	{
		return m_MeshLoader.GetMesh();
	}
}