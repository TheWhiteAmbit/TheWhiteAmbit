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




#include "DirectX10Texture.h"
namespace TheWhiteAmbit {

	DirectX10Texture::DirectX10Texture(DirectX10Renderer* a_pRenderer)
	{
		m_pTexture=NULL;
		m_pRenderer=a_pRenderer;

		D3D10_TEXTURE2D_DESC desc;
		ZeroMemory( &desc, sizeof(desc) );
		desc.Width = 16;
		desc.Height = 16;
		desc.MipLevels = 1;
		desc.ArraySize = 1;
		desc.Format = DXGI_FORMAT_R8G8B8A8_UINT;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.Usage = D3D10_USAGE_DEFAULT;
		//desc.BindFlags = D3D10_BIND_DEPTH_STENCIL;
		desc.BindFlags = D3D10_BIND_RENDER_TARGET | D3D10_BIND_SHADER_RESOURCE;
		//desc.BindFlags = bindFlags;
		desc.CPUAccessFlags = 0;
		desc.MiscFlags = 0;
		desc.CPUAccessFlags = 0;
		desc.MiscFlags = 0;
		HRESULT hr = m_pRenderer->getDevice()->CreateTexture2D( &desc, NULL, &m_pTexture );
		if( FAILED(hr) )
			return;
	}

	DirectX10Texture::DirectX10Texture(DirectX10Renderer* a_pRenderer, unsigned int width, unsigned int height, D3D10_USAGE usage, DXGI_FORMAT format, UINT bindFlags)
	{
		m_pTexture=NULL;
		m_pRenderer=a_pRenderer;


		D3D10_TEXTURE2D_DESC desc;
		ZeroMemory( &desc, sizeof(desc) );
		desc.Width = width;
		desc.Height = height;
		desc.MipLevels = 1;
		desc.ArraySize = 1;
		desc.Format = format;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.Usage = usage;
		//desc.BindFlags = D3D10_BIND_DEPTH_STENCIL;
		//desc.BindFlags = D3D10_BIND_RENDER_TARGET | D3D10_BIND_SHADER_RESOURCE;
		desc.BindFlags = bindFlags;
		desc.CPUAccessFlags = 0;
		desc.MiscFlags = 0;
		desc.CPUAccessFlags = 0;
		desc.MiscFlags = 0;
		HRESULT hr = m_pRenderer->getDevice()->CreateTexture2D( &desc, NULL, &m_pTexture );
		if( FAILED(hr) )
			return;
	}

	DirectX10Texture::~DirectX10Texture(void)
	{
	}

	ID3D10Texture2D* DirectX10Texture::getTexture(void)
	{
		return m_pTexture;
	}

	Grid<Color>* DirectX10Texture::getGrid(void)
	{
		return NULL;
	}

	void DirectX10Texture::setGrid(Grid<Color>* a_pGrid)
	{
		//TODO: implement this
	}

	unsigned int DirectX10Texture::getWidth(void)
	{
		D3D10_TEXTURE2D_DESC desc;
		ZeroMemory( &desc, sizeof(desc) );
		m_pTexture->GetDesc(&desc);
		return desc.Width;
	}

	unsigned int DirectX10Texture::getHeight(void)
	{
		D3D10_TEXTURE2D_DESC desc;
		ZeroMemory( &desc, sizeof(desc) );
		m_pTexture->GetDesc(&desc);
		return desc.Height;
	}	

	void DirectX10Texture::fillColorGardient(void)
	{
		//TODO: implement this

	}
}