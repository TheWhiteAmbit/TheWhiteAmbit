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




#include "DirectX9Effect.h"
namespace TheWhiteAmbit {

	DirectX9Effect::DirectX9Effect(DirectX9Renderer* a_pRenderer, LPCWSTR a_sFilenameEffect)
	{
		m_pRenderer=a_pRenderer;
		if(FAILED( D3DXCreateEffectFromFile( m_pRenderer->getDevice(), a_sFilenameEffect, NULL, NULL, D3DXFX_NOT_CLONEABLE | D3DXFX_LARGEADDRESSAWARE, NULL, &m_pEffect, NULL )))
		{
			this->m_pEffect=NULL;
			MessageBox( NULL,
				L"The FX file cannot be created.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
			return;
		}
	}

	DirectX9Effect::~DirectX9Effect(void)
	{
		if(m_pEffect)
			m_pEffect->Release();
		m_pEffect=NULL;
	}

	ID3DXEffect*    DirectX9Effect::getEffect(void)
	{
		return this->m_pEffect;
	}

	void*	DirectX9Effect::getDirectX9Effect(void){
		return (DirectX9Effect*)this;
	}

	//TODO: insert effect code
	void*	DirectX9Effect::getDirectX10Effect(void){
		return NULL;
	}

	void DirectX9Effect::setValue(LPCWSTR name, double a_fValue)
	{
		if(m_pEffect)
		{	
			char nameChar[MAX_PATH];
			WideCharToMultiByte( CP_ACP, 0, name, -1, (LPSTR)&nameChar, MAX_PATH, NULL, NULL );		

			//m_pEffect->SetFloat(nameChar, (FLOAT)a_fValue);
			D3DXHANDLE ptxVariable;
			ptxVariable= m_pEffect->GetParameterByName( NULL, nameChar );
			if(ptxVariable)
				m_pEffect->SetFloat(ptxVariable, (FLOAT)a_fValue);
		}
	}

	void DirectX9Effect::setValue(LPCWSTR name, D3DXMATRIX* a_fValue)
	{
		if(m_pEffect)
		{	
			char nameChar[MAX_PATH];
			WideCharToMultiByte( CP_ACP, 0, name, -1, (LPSTR)&nameChar, MAX_PATH, NULL, NULL );		

			//m_pEffect->SetFloat(nameChar, (FLOAT)a_fValue);
			D3DXHANDLE ptxVariable;
			ptxVariable= m_pEffect->GetParameterByName( NULL, nameChar );
			if(ptxVariable)
				m_pEffect->SetMatrix(ptxVariable, a_fValue);
		}
	}

	void DirectX9Effect::setValue(LPCWSTR name, DirectX9Texture* a_pTexture)
	{
		if(m_pEffect)
		{	
			char nameChar[MAX_PATH];
			WideCharToMultiByte( CP_ACP, 0, name, -1, (LPSTR)&nameChar, MAX_PATH, NULL, NULL );		

			//m_pEffect->SetTexture(nameChar, a_pTexture->getTexture());
			D3DXHANDLE ptxVariable;
			ptxVariable= m_pEffect->GetParameterByName( NULL, nameChar );
			if(ptxVariable)
				m_pEffect->SetTexture( ptxVariable, a_pTexture->getTexture() );

		}
	}
}