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




#include "DirectX10Effect.h"

namespace TheWhiteAmbit {
	DirectX10Effect::DirectX10Effect(DirectX10Renderer* a_pRenderer, LPCWSTR a_sFilenameEffect)
	{
		m_pRenderer=a_pRenderer;
		m_pEffect=NULL;

		HRESULT hr = S_OK;
		// Create the effect

		//DWORD dwShaderFlags = D3D10_SHADER_ENABLE_STRICTNESS;
		DWORD dwShaderFlags = 0;
		hr = D3DX10CreateEffectFromFile( a_sFilenameEffect, NULL, NULL, "fx_4_0", dwShaderFlags, 0,
			m_pRenderer->getDevice(), NULL, NULL, &m_pEffect, NULL, NULL );
		if( FAILED( hr ) )
		{
			m_pEffect=NULL;
			MessageBox( NULL,
				L"The FX file cannot be created.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
		}
	}

	DirectX10Effect::~DirectX10Effect(void)
	{
		if( m_pEffect ) m_pEffect->Release();
	}

	ID3D10Effect*	DirectX10Effect::getEffect(void)
	{
		return m_pEffect;
	}
}