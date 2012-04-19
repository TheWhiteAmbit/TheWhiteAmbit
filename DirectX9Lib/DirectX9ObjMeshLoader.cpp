//--------------------------------------------------------------------------------------
// File: MeshLoader.cpp
//
// Wrapper class for ID3DXMesh interface. Handles loading mesh data from an .obj file
// and resource management for material textures.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#pragma warning(disable: 4995)
#include "DirectX9ObjMeshLoader.h"
#include <fstream>
using namespace std;
#pragma warning(default: 4995)

namespace TheWhiteAmbit {


	//--------------------------------------------------------------------------------------
	CMeshLoader::CMeshLoader()
	{
		m_pd3dDevice = NULL;
		m_pMesh = NULL;

		ZeroMemory( m_strMediaDir, sizeof( m_strMediaDir ) );
		//NeedCurrentDirectoryForExePath((LPCWSTR)&m_strMediaDir);
	}


	//--------------------------------------------------------------------------------------
	CMeshLoader::~CMeshLoader()
	{
		Destroy();
	}


	//--------------------------------------------------------------------------------------
	void CMeshLoader::Destroy()
	{
		for( int iMaterial = 0; iMaterial < m_Materials.GetSize(); iMaterial++ )
		{
			Material* pMaterial = m_Materials.GetAt( iMaterial );

			// Avoid releasing the same texture twice
			for( int x = iMaterial + 1; x < m_Materials.GetSize(); x++ )
			{
				Material* pCur = m_Materials.GetAt( x );
				if( pCur->pTextureDiffuse == pMaterial->pTextureDiffuse )
					pCur->pTextureDiffuse = NULL;
			}

			SAFE_RELEASE( pMaterial->pTextureDiffuse );
			SAFE_DELETE( pMaterial );
		}

		m_Materials.RemoveAll();
		m_Vertices.RemoveAll();
		m_Indices.RemoveAll();
		m_Attributes.RemoveAll();

		SAFE_RELEASE( m_pMesh );
		m_pd3dDevice = NULL;
	}


	//--------------------------------------------------------------------------------------
	HRESULT CMeshLoader::Create( IDirect3DDevice9* pd3dDevice, const WCHAR* strFilename )
	{
		HRESULT hr;
		//WCHAR str[ MAX_PATH ] = {0};

		// Start clean
		Destroy();

		// Store the device pointer
		m_pd3dDevice = pd3dDevice;

		// Load the vertex buffer, index buffer, and subset information from a file. In this case, 
		// an .obj file was chosen for simplicity, but it's meant to illustrate that ID3DXMesh objects
		// can be filled from any mesh file format once the necessary data is extracted from file.
		V_RETURN( LoadGeometryFromOBJ( strFilename ) );

		// Set the current directory based on where the mesh was found
		WCHAR wstrOldDir[MAX_PATH] = {0};
		GetCurrentDirectory( MAX_PATH, wstrOldDir );
		SetCurrentDirectory( m_strMediaDir );

		// Load material textures
		for( int iMaterial = 0; iMaterial < m_Materials.GetSize(); iMaterial++ )
		{
			Material* pMaterial = m_Materials.GetAt( iMaterial );
			if( pMaterial->strTextureDiffuse[0] )
			{
				// Avoid loading the same texture twice
				bool bFound = false;
				for( int x = 0; x < iMaterial; x++ )
				{
					Material* pCur = m_Materials.GetAt( x );
					if( 0 == wcscmp( pCur->strTextureDiffuse, pMaterial->strTextureDiffuse ) )
					{
						bFound = true;
						pMaterial->pTextureDiffuse = pCur->pTextureDiffuse;
						break;
					}
				}

				// Not found, load the texture
				if( !bFound )
				{
					//V_RETURN( DXUTFindDXSDKMediaFileCch( str, MAX_PATH, pMaterial->strTextureDiffuse ) );
					V_RETURN( D3DXCreateTextureFromFile( pd3dDevice, pMaterial->strTextureDiffuse,
						&( pMaterial->pTextureDiffuse ) ) );
				}
			}        
			if( pMaterial->strTextureBump[0] )
			{
				// Avoid loading the same texture twice
				bool bFound = false;
				for( int x = 0; x < iMaterial; x++ )
				{
					Material* pCur = m_Materials.GetAt( x );
					if( 0 == wcscmp( pCur->strTextureBump, pMaterial->strTextureBump ) )
					{
						bFound = true;
						pMaterial->pTextureBump = pCur->pTextureBump;
						break;
					}
				}

				// Not found, load the texture
				if( !bFound )
				{
					//V_RETURN( DXUTFindDXSDKMediaFileCch( str, MAX_PATH, pMaterial->strTextureDiffuse ) );
					V_RETURN( D3DXCreateTextureFromFile( pd3dDevice, pMaterial->strTextureBump,
						&( pMaterial->pTextureBump ) ) );
				}
			}
		}

		// Restore the original current directory
		SetCurrentDirectory( wstrOldDir );

		// Create the encapsulated mesh
		ID3DXMesh* pMesh = NULL;
		V_RETURN( D3DXCreateMesh( m_Indices.GetSize() / 3, m_Vertices.GetSize(),
			D3DXMESH_MANAGED | D3DXMESH_32BIT, Vertex::getVertexElements(),
			pd3dDevice, &pMesh ) );

		// Copy the vertex data
		Vertex* pVertex;
		V_RETURN( pMesh->LockVertexBuffer( 0, ( void** )&pVertex ) );
		memcpy( pVertex, m_Vertices.GetData(), m_Vertices.GetSize() * sizeof( Vertex ) );
		pMesh->UnlockVertexBuffer();
		m_Vertices.RemoveAll();

		// Copy the index data
		DWORD* pIndex;
		V_RETURN( pMesh->LockIndexBuffer( 0, ( void** )&pIndex ) );
		memcpy( pIndex, m_Indices.GetData(), m_Indices.GetSize() * sizeof( DWORD ) );
		pMesh->UnlockIndexBuffer();
		m_Indices.RemoveAll();

		// Copy the attribute data
		DWORD* pSubset;
		V_RETURN( pMesh->LockAttributeBuffer( 0, &pSubset ) );
		memcpy( pSubset, m_Attributes.GetData(), m_Attributes.GetSize() * sizeof( DWORD ) );
		pMesh->UnlockAttributeBuffer();
		m_Attributes.RemoveAll();

		// Reorder the vertices according to subset and optimize the mesh for this graphics 
		// card's vertex cache. When rendering the mesh's triangle list the vertices will 
		// cache hit more often so it won't have to re-execute the vertex shader.
		DWORD* aAdjacency = new DWORD[pMesh->GetNumFaces() * 3];
		if( aAdjacency == NULL )
			return E_OUTOFMEMORY;

		V( pMesh->GenerateAdjacency( 1e-6f, aAdjacency ) );
		V( pMesh->OptimizeInplace( D3DXMESHOPT_ATTRSORT | D3DXMESHOPT_VERTEXCACHE, aAdjacency, NULL, NULL, NULL ) );

		SAFE_DELETE_ARRAY( aAdjacency );
		m_pMesh = pMesh;

		return S_OK;
	}


	//--------------------------------------------------------------------------------------
	HRESULT CMeshLoader::LoadGeometryFromOBJ( const WCHAR* strFileName )
	{
		WCHAR strMaterialFilename[MAX_PATH] = {0};
		//WCHAR wstr[MAX_PATH];
		char str[MAX_PATH];
		HRESULT hr;

		// Find the file
		//V_RETURN( DXUTFindDXSDKMediaFileCch( wstr, MAX_PATH, strFileName ) );
		WideCharToMultiByte( CP_ACP, 0, strFileName, -1, str, MAX_PATH, NULL, NULL );

		// Store the directory where the mesh was found
		StringCchCopy( m_strMediaDir, MAX_PATH - 1, strFileName );
		WCHAR* pch = wcsrchr( m_strMediaDir, L'\\' );
		if( pch )
			*pch = NULL;

		// Create temporary storage for the input data. Once the data has been loaded into
		// a reasonable format we can create a D3DXMesh object and load it with the mesh data.
		CGrowableArray <D3DXVECTOR4> Positions;
		CGrowableArray <D3DXVECTOR2> TexCoords;
		CGrowableArray <D3DXVECTOR4> Normals;

		// The first subset uses the default material
		Material* pMaterial = new Material();
		if( pMaterial == NULL )
			return E_OUTOFMEMORY;

		InitMaterial( pMaterial );
		StringCchCopy( pMaterial->strName, MAX_PATH - 1, L"default" );
		m_Materials.Add( pMaterial );

		DWORD dwCurSubset = 0;

		// File input
		WCHAR strCommand[256] = {0};
		wifstream InFile( str );
		if( !InFile )
			return DXTRACE_ERR( L"wifstream::open", E_FAIL );

		for(; ; )
		{
			InFile >> strCommand;
			if( !InFile )
				break;

			if( 0 == wcscmp( strCommand, L"#" ) )
			{
				// Comment
			}
			else if( 0 == wcscmp( strCommand, L"v" ) )
			{
				// Vertex Position
				float x, y, z;
				InFile >> x >> y >> z;
				Positions.Add( D3DXVECTOR4( x, y, z, 1 ) );
			}
			else if( 0 == wcscmp( strCommand, L"vt" ) )
			{
				// Vertex TexCoord
				float u, v;
				InFile >> u >> v;
				TexCoords.Add( D3DXVECTOR2( u, v ) );
			}
			else if( 0 == wcscmp( strCommand, L"vn" ) )
			{
				// Vertex Normal
				float x, y, z;
				InFile >> x >> y >> z;
				Normals.Add( D3DXVECTOR4( x, y, z, 0) );
			}
			else if( 0 == wcscmp( strCommand, L"f" ) )
			{	
				// Face
				UINT iPosition, iTexCoord, iNormal;
				Vertex vertex;

				DWORD indexQuad[9];			

				for( UINT iFace = 0; iFace < 4; iFace++ )
				{
					//find if we have no quad, if so break
					if( 3==iFace ) {
						wchar_t peek;
						do{
							InFile.get(peek);
						}while(peek==L' ');
						InFile.putback(peek);
						if(peek==10 || peek==13){
							m_Indices.Add( indexQuad[0] );
							m_Indices.Add( indexQuad[1] );
							m_Indices.Add( indexQuad[2] );
							break;				
						}
					}

					ZeroMemory( &vertex, sizeof( Vertex ) );				

					// OBJ format uses 1-based arrays
					InFile >> iPosition;

					vertex.pos = Positions[ iPosition - 1 ];

					if( '/' == InFile.peek() )
					{
						InFile.ignore();

						if( '/' != InFile.peek() )
						{
							// Optional texture coordinate
							InFile >> iTexCoord;
							vertex.tex = TexCoords[ iTexCoord - 1 ];
						}

						if( '/' == InFile.peek() )
						{
							InFile.ignore();

							// Optional vertex normal
							InFile >> iNormal;
							vertex.norm = Normals[ iNormal - 1 ];
						}
					}

					// If a duplicate vertex doesn't exist, add this vertex to the Vertices
					// list. Store the index in the Indices array. The Vertices and Indices
					// lists will eventually become the Vertex Buffer and Index Buffer for
					// the mesh.
					DWORD index = AddVertex( iPosition, &vertex );

					//remember indices for quad if we find one above
					//then add the missing indices, else this wont be reached again
					switch(iFace){
					case 0: 
						indexQuad[0]=index;
						break;
					case 1: 
						indexQuad[1]=index;
						break;
					case 2: 
						indexQuad[2]=index;
						break;
					case 3:
						indexQuad[3]=index;

						//calculate new vertex at center, so there will be
						//no texture skewing in quads this way

						Vertex vertex01;						
						ZeroMemory( &vertex01, sizeof( Vertex ) );													
						vertex01.pos  = (m_Vertices[indexQuad[0]].pos
							+ m_Vertices[indexQuad[1]].pos)/2.0f;
						vertex01.norm  = (m_Vertices[indexQuad[0]].norm
							+ m_Vertices[indexQuad[1]].norm)/2.0f;
						vertex01.tex  = (m_Vertices[indexQuad[0]].tex
							+ m_Vertices[indexQuad[1]].tex)/2.0f;

						Vertex vertex12;						
						ZeroMemory( &vertex12, sizeof( Vertex ) );													
						vertex12.pos  = (m_Vertices[indexQuad[1]].pos
							+ m_Vertices[indexQuad[2]].pos)/2.0f;
						vertex12.norm  = (m_Vertices[indexQuad[1]].norm
							+ m_Vertices[indexQuad[2]].norm)/2.0f;
						vertex12.tex  = (m_Vertices[indexQuad[1]].tex
							+ m_Vertices[indexQuad[2]].tex)/2.0f;

						Vertex vertex23;						
						ZeroMemory( &vertex23, sizeof( Vertex ) );													
						vertex23.pos  = (m_Vertices[indexQuad[2]].pos
							+ m_Vertices[indexQuad[3]].pos)/2.0f;
						vertex23.norm  = (m_Vertices[indexQuad[2]].norm
							+ m_Vertices[indexQuad[3]].norm)/2.0f;
						vertex23.tex  = (m_Vertices[indexQuad[2]].tex
							+ m_Vertices[indexQuad[3]].tex)/2.0f;

						Vertex vertex30;						
						ZeroMemory( &vertex30, sizeof( Vertex ) );													
						vertex30.pos  = (m_Vertices[indexQuad[3]].pos
							+ m_Vertices[indexQuad[0]].pos)/2.0f;
						vertex30.norm  = (m_Vertices[indexQuad[3]].norm
							+ m_Vertices[indexQuad[0]].norm)/2.0f;
						vertex30.tex  = (m_Vertices[indexQuad[3]].tex
							+ m_Vertices[indexQuad[0]].tex)/2.0f;

						ZeroMemory( &vertex, sizeof( Vertex ) );
						vertex.pos  = (vertex01.pos
							+ vertex12.pos
							+ vertex23.pos
							+ vertex30.pos)/4.0f;
						vertex.norm  = (vertex01.norm
							+ vertex12.norm
							+ vertex23.norm
							+ vertex30.norm)/4.0f;
						vertex.tex  = (vertex01.tex
							+ vertex12.tex
							+ vertex23.tex
							+ vertex30.tex)/4.0f;

						DWORD index0123 = AddVertex( iPosition, &vertex, true );
						DWORD index01 = AddVertex( iPosition, &vertex01, true );
						DWORD index12 = AddVertex( iPosition, &vertex12, true );
						DWORD index23 = AddVertex( iPosition, &vertex23, true );
						DWORD index30 = AddVertex( iPosition, &vertex30, true );

						m_Indices.Add( indexQuad[0] );
						m_Indices.Add( index01 );
						m_Indices.Add( index0123 );

						m_Indices.Add( index0123 );
						m_Indices.Add( index01 );
						m_Indices.Add( indexQuad[1] );

						m_Indices.Add( indexQuad[1] );
						m_Indices.Add( index12 );
						m_Indices.Add( index0123 );

						m_Indices.Add( index0123 );
						m_Indices.Add( index12 );
						m_Indices.Add( indexQuad[2] );

						m_Indices.Add( indexQuad[2] );
						m_Indices.Add( index23 );
						m_Indices.Add( index0123 );

						m_Indices.Add( index0123 );
						m_Indices.Add( index23 );
						m_Indices.Add( indexQuad[3] );

						m_Indices.Add( indexQuad[3] );
						m_Indices.Add( index30 );
						m_Indices.Add( index0123 );

						m_Indices.Add( index0123 );
						m_Indices.Add( index30 );
						m_Indices.Add( indexQuad[0] );

						/////////////////////////////////
						//						m_Indices.Add( indexQuad[0] );
						//						m_Indices.Add( indexQuad[1] );
						//						m_Indices.Add( index );
						//
						//						m_Indices.Add( index );
						//						m_Indices.Add( indexQuad[1] );
						//						m_Indices.Add( indexQuad[2] );
						//
						//						m_Indices.Add( indexQuad[2] );
						//						m_Indices.Add( indexQuad[3] );
						//						m_Indices.Add( index );
						//
						//						m_Indices.Add( index );
						//						m_Indices.Add( indexQuad[3] );
						//						m_Indices.Add( indexQuad[0] );
						break;
					}

				}
				m_Attributes.Add( dwCurSubset );
			}
			else if( 0 == wcscmp( strCommand, L"mtllib" ) )
			{
				// Material library
				InFile >> strMaterialFilename;
			}
			else if( 0 == wcscmp( strCommand, L"usemtl" ) )
			{
				// Material
				WCHAR strName[MAX_PATH] = {0};
				InFile >> strName;

				bool bFound = false;
				for( int iMaterial = 0; iMaterial < m_Materials.GetSize(); iMaterial++ )
				{
					Material* pCurMaterial = m_Materials.GetAt( iMaterial );
					if( 0 == wcscmp( pCurMaterial->strName, strName ) )
					{
						bFound = true;
						dwCurSubset = iMaterial;
						break;
					}
				}

				if( !bFound )
				{
					pMaterial = new Material();
					if( pMaterial == NULL )
						return E_OUTOFMEMORY;

					dwCurSubset = m_Materials.GetSize();

					InitMaterial( pMaterial );
					StringCchCopy( pMaterial->strName, MAX_PATH - 1, strName );

					m_Materials.Add( pMaterial );
				}
			}
			else
			{
				// Unimplemented or unrecognized command
			}

			InFile.ignore( 1000, '\n' );
		}

		// Cleanup
		InFile.close();
		DeleteCache();

		//TODO: read materials somewhere else

		//// If an associated material file was found, read that in as well.
		//if( strMaterialFilename[0] )
		//{
		//    V_RETURN( LoadMaterialsFromMTL( strMaterialFilename ) );
		//}

		return S_OK;
	}


	//--------------------------------------------------------------------------------------
	DWORD CMeshLoader::AddVertex( UINT hash, Vertex* pVertex, bool forceNewEntry )
	{
		// If this vertex doesn't already exist in the Vertices list, create a new entry.
		// Add the index of the vertex to the Indices list.
		bool bFoundInList = false;
		DWORD index = 0;

		// Since it's very slow to check every element in the vertex list, a hashtable stores
		// vertex indices according to the vertex position's index as reported by the OBJ file
		if( ( UINT )m_VertexCache.GetSize() > hash && !forceNewEntry)
		{
			CacheEntry* pEntry = m_VertexCache.GetAt( hash );
			while( pEntry != NULL )
			{
				Vertex* pCacheVertex = m_Vertices.GetData() + pEntry->index;

				// If this vertex is identical to the vertex already in the list, simply
				// point the index buffer to the existing vertex
				if( 0 == memcmp( pVertex, pCacheVertex, sizeof( Vertex ) ) )
				{
					bFoundInList = true;
					index = pEntry->index;
					break;
				}

				pEntry = pEntry->pNext;
			}
		}

		// Vertex was not found in the list. Create a new entry, both within the Vertices list
		// and also within the hashtable cache
		if( !bFoundInList )
		{
			// Add to the Vertices list
			index = m_Vertices.GetSize();
			m_Vertices.Add( *pVertex );

			// Add this to the hashtable
			CacheEntry* pNewEntry = new CacheEntry;
			if( pNewEntry == NULL )
				return E_OUTOFMEMORY;

			pNewEntry->index = index;
			pNewEntry->pNext = NULL;

			// Grow the cache if needed
			while( ( UINT )m_VertexCache.GetSize() <= hash )
			{
				m_VertexCache.Add( NULL );
			}

			// Add to the end of the linked list
			CacheEntry* pCurEntry = m_VertexCache.GetAt( hash );
			if( pCurEntry == NULL )
			{
				// This is the head element
				m_VertexCache.SetAt( hash, pNewEntry );
			}
			else
			{
				// Find the tail
				while( pCurEntry->pNext != NULL )
				{
					pCurEntry = pCurEntry->pNext;
				}

				pCurEntry->pNext = pNewEntry;
			}
		}

		return index;
	}


	//--------------------------------------------------------------------------------------
	void CMeshLoader::DeleteCache()
	{
		// Iterate through all the elements in the cache and subsequent linked lists
		for( int i = 0; i < m_VertexCache.GetSize(); i++ )
		{
			CacheEntry* pEntry = m_VertexCache.GetAt( i );
			while( pEntry != NULL )
			{
				CacheEntry* pNext = pEntry->pNext;
				SAFE_DELETE( pEntry );
				pEntry = pNext;
			}
		}

		m_VertexCache.RemoveAll();
	}


	//--------------------------------------------------------------------------------------
	HRESULT CMeshLoader::LoadMaterialsFromMTL( const WCHAR* strFileName )
	{
		//HRESULT hr;

		// Set the current directory based on where the mesh was found
		WCHAR wstrOldDir[MAX_PATH] = {0};
		GetCurrentDirectory( MAX_PATH, wstrOldDir );
		SetCurrentDirectory( m_strMediaDir );

		// Find the file
		//WCHAR strPath[MAX_PATH];
		char cstrPath[MAX_PATH];
		//V_RETURN( DXUTFindDXSDKMediaFileCch( strPath, MAX_PATH, strFileName ) );
		WideCharToMultiByte( CP_ACP, 0, strFileName, -1, cstrPath, MAX_PATH, NULL, NULL );

		// File input
		WCHAR strCommand[256] = {0};
		wifstream InFile( cstrPath );
		if( !InFile )
			return DXTRACE_ERR( L"wifstream::open", E_FAIL );

		// Restore the original current directory
		SetCurrentDirectory( wstrOldDir );

		Material* pMaterial = NULL;

		for(; ; )
		{
			InFile >> strCommand;
			if( !InFile )
				break;

			if( 0 == wcscmp( strCommand, L"newmtl" ) )
			{
				// Switching active materials
				WCHAR strName[MAX_PATH] = {0};
				InFile >> strName;

				pMaterial = NULL;
				for( int i = 0; i < m_Materials.GetSize(); i++ )
				{
					Material* pCurMaterial = m_Materials.GetAt( i );
					if( 0 == wcscmp( pCurMaterial->strName, strName ) )
					{
						pMaterial = pCurMaterial;
						break;
					}
				}
			}

			// The rest of the commands rely on an active material
			if( pMaterial == NULL )
				continue;

			if( 0 == wcscmp( strCommand, L"#" ) )
			{
				// Comment
			}
			else if( 0 == wcscmp( strCommand, L"Ka" ) )
			{
				// Ambient color
				float r, g, b;
				InFile >> r >> g >> b;
				pMaterial->vAmbient = D3DXVECTOR3( r, g, b );
			}
			else if( 0 == wcscmp( strCommand, L"Kd" ) )
			{
				// Diffuse color
				float r, g, b;
				InFile >> r >> g >> b;
				pMaterial->vDiffuse = D3DXVECTOR3( r, g, b );
			}
			else if( 0 == wcscmp( strCommand, L"Ks" ) )
			{
				// Specular color
				float r, g, b;
				InFile >> r >> g >> b;
				pMaterial->vSpecular = D3DXVECTOR3( r, g, b );
			}
			else if( 0 == wcscmp( strCommand, L"d" ) ||
				0 == wcscmp( strCommand, L"Tr" ) )
			{
				// Alpha
				InFile >> pMaterial->fAlpha;
			}
			else if( 0 == wcscmp( strCommand, L"Ns" ) )
			{
				// Shininess
				int nShininess;
				InFile >> nShininess;
				pMaterial->nShininess = nShininess;
			}
			else if( 0 == wcscmp( strCommand, L"illum" ) )
			{
				// Specular on/off
				int illumination;
				InFile >> illumination;
				pMaterial->bSpecular = ( illumination == 2 );
			}
			else if( 0 == wcscmp( strCommand, L"map_Kd" ) )
			{
				// Texture
				InFile >> pMaterial->strTextureDiffuse;
			}
			else if( 0 == wcscmp( strCommand, L"bump" ) )
			{
				// Texture
				InFile >> pMaterial->strTextureBump;
			}

			else
			{
				// Unimplemented or unrecognized command
			}

			InFile.ignore( 1000, L'\n' );
		}

		InFile.close();

		return S_OK;
	}


	//--------------------------------------------------------------------------------------
	void CMeshLoader::InitMaterial( Material* pMaterial )
	{
		ZeroMemory( pMaterial, sizeof( Material ) );

		pMaterial->vAmbient = D3DXVECTOR3( 0.2f, 0.2f, 0.2f );
		pMaterial->vDiffuse = D3DXVECTOR3( 0.8f, 0.8f, 0.8f );
		pMaterial->vSpecular = D3DXVECTOR3( 1.0f, 1.0f, 1.0f );
		pMaterial->nShininess = 0;
		pMaterial->fAlpha = 1.0f;
		pMaterial->bSpecular = false;
		pMaterial->pTextureDiffuse = NULL;
	}

}