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




#include "DirectX9Texture.h"
#include "../CudaLib/rendering.h"

namespace TheWhiteAmbit {
	DirectX9Texture::DirectX9Texture(DirectX9Renderer* a_pRenderer)
	{
		m_pRenderer=a_pRenderer;

		m_iWidth=(unsigned int)RESOLUTION_X;
		m_iHeight=(unsigned int)RESOLUTION_Y;
		//m_iWidth=512;
		//m_iHeight=512;
		m_pTexture=NULL;
		m_pRenderer->getDevice()->CreateTexture(
			m_iWidth,
			m_iHeight,
			1,
			D3DUSAGE_RENDERTARGET,
			//D3DUSAGE_DYNAMIC,
			//D3DFMT_A8R8G8B8,
			D3DFMT_A32B32G32R32F,
			D3DPOOL_DEFAULT,
			&m_pTexture,
			NULL);

		//D3DSURFACE_DESC desc;
		//m_pTexture->GetLevelDesc(0, &desc);
		//m_iWidth=desc.Width;
		//m_iHeight=desc.Height;
		//this->fillColorGardient();
	}

	DirectX9Texture::DirectX9Texture(DirectX9Renderer* a_pRenderer, unsigned int width, unsigned int height, DWORD usage, D3DFORMAT a_hFormat)
	{
		m_pRenderer=a_pRenderer;
		m_iWidth=width;
		m_iHeight=height;
		m_pTexture=NULL;
		HRESULT hr = m_pRenderer->getDevice()->CreateTexture(
			m_iWidth,
			m_iHeight,
			1,
			usage,
			a_hFormat,
			D3DPOOL_DEFAULT,
			&m_pTexture,
			NULL);

		//D3DSURFACE_DESC desc;
		//m_pTexture->GetLevelDesc(0, &desc);
		//m_iWidth=desc.Width;
		//m_iHeight=desc.Height;
	}

	DirectX9Texture::~DirectX9Texture(void)
	{
		if(m_pTexture)
			m_pTexture->Release();
		m_pTexture=NULL;
	}

	IDirect3DTexture9* DirectX9Texture::getTexture(void)
	{
		return this->m_pTexture;
	}

	IDirect3DSurface9* DirectX9Texture::getSurface(unsigned int level)
	{
		IDirect3DSurface9* pRenderSurface=NULL;
		if(m_pTexture)
			m_pTexture->GetSurfaceLevel(level,&pRenderSurface);
		return pRenderSurface;
	}

	static void FillColorGardient(D3DXVECTOR4* pOut, const D3DXVECTOR2* pTexCoord,  const D3DXVECTOR2* pTexelSize, LPVOID ptr)
	{
		*pOut=D3DXVECTOR4(pTexCoord->x,
			pTexCoord->y,
			((1.0f-pTexCoord->x)*(1.0f-pTexCoord->y)),
			.5f);
	}

	void DirectX9Texture::fillColorGardient(void)
	{	
		//D3DXFillTexture(m_pTexture, (LPD3DXFILL2D) &FillColorGardient, NULL);

		//Grid<Color>* pGrid=new Grid<Color>(m_iWidth, m_iHeight);
		//for(unsigned int x=0; x<m_iWidth; x++)
		//{
		//	for(unsigned int y=0; y<m_iHeight; y++)
		//	{
		//		unsigned char pixel[4]={x%256, y%256, 0, 0};
		//		pGrid->setPixel(x, y, pixel);
		//	}
		//}

		//Grid<Color>* pGrid=new Grid<Color>(m_iWidth, m_iHeight);
		//for(unsigned int y=0; y<m_iHeight; y++)
		//{
		//	for(unsigned int x=0; x<m_iWidth; x++)
		//	{
		//		unsigned int val=((x+y)%2)*255;
		//		D3DXVECTOR4 pixel=D3DXVECTOR4((float)val, (float)val, (float)val, (float)val);
		//		pGrid->setPixel(x, y, pixel);
		//	}
		//}

		Grid<unsigned char[4]>* pGrid=new Grid<unsigned char[4]>(m_iWidth, m_iHeight);
		for(unsigned int y=0; y<m_iHeight; y++)
		{
			for(unsigned int x=0; x<m_iWidth; x++)
			{
				unsigned int val=((x+y)%2)*255;
				unsigned char pixel[4];
				pixel[0]=(unsigned char)val;
				pixel[1]=(unsigned char)val;
				pixel[2]=(unsigned char)val;
				pixel[3]=(unsigned char)val;
				pGrid->setPixel(x, y, pixel);
			}
		}

		this->setGrid(pGrid);
		delete pGrid;
	}

	Grid<Color>* DirectX9Texture::getGrid(void)
	{
		if(!m_pTexture)
			return NULL;

		HRESULT hr=S_OK;
		Grid<Color>* pGrid=NULL;
		D3DSURFACE_DESC desc;
		m_pTexture->GetLevelDesc(0, &desc);
		//TODO: get texture format from color
		if(D3DFMT_A32B32G32R32F==desc.Format)
		{
			IDirect3DSurface9* offscreenSurface;
			hr = m_pRenderer->getDevice()->CreateOffscreenPlainSurface( desc.Width, desc.Height, desc.Format, D3DPOOL_SYSTEMMEM, &offscreenSurface, NULL );
			if( FAILED(hr) )
				return NULL;

			IDirect3DSurface9* sourceSurface;
			m_pTexture->GetSurfaceLevel(0, &sourceSurface);
			hr = m_pRenderer->getDevice()->GetRenderTargetData(sourceSurface, offscreenSurface );

			m_iWidth=desc.Width;
			m_iHeight=desc.Height;

			D3DLOCKED_RECT lockedRect;
			if(!FAILED(offscreenSurface->LockRect(&lockedRect, NULL, D3DLOCK_READONLY)))
			{
				pGrid=new Grid<Color>(m_iWidth, m_iHeight);
				for(unsigned int y=0; y<m_iHeight; y++)
				{
					void* src=((char*)lockedRect.pBits)+(y*lockedRect.Pitch);
					memcpy(pGrid->getRawRowData(y), src, pGrid->getRawRowDataByteCount());
				}
				offscreenSurface->UnlockRect();			
			}		
			offscreenSurface->Release();
		}
		return pGrid;
	}

	void DirectX9Texture::setGrid(Grid<unsigned char[4]>* a_pGrid)
	{
		HRESULT hr=S_OK;
		Grid<Color>* pGrid=NULL;
		D3DSURFACE_DESC desc;
		m_pTexture->GetLevelDesc(0, &desc);
		//TODO: get texture format from color when 8 bit color
		if(D3DFMT_A8R8G8B8==desc.Format)
		{
			IDirect3DSurface9* offscreenSurface;
			hr = m_pRenderer->getDevice()->CreateOffscreenPlainSurface( desc.Width, desc.Height, desc.Format, D3DPOOL_SYSTEMMEM, &offscreenSurface, NULL );
			if( FAILED(hr) )
				return;

			IDirect3DSurface9* destinationSurface;
			m_pTexture->GetSurfaceLevel(0, &destinationSurface);

			m_iWidth=desc.Width;
			m_iHeight=desc.Height;

			D3DLOCKED_RECT lockedRect;
			HRESULT hr=S_OK;
			if(!FAILED(offscreenSurface->LockRect(&lockedRect, NULL, D3DLOCK_DISCARD)))
			{
				for(unsigned int y=0; y<m_iHeight; y++)
				{
					void* dst=((char*)lockedRect.pBits)+(y*lockedRect.Pitch);
					memcpy(dst, a_pGrid->getRawRowData(y), a_pGrid->getRawRowDataByteCount());
				}
				offscreenSurface->UnlockRect();
			}
			m_pRenderer->getDevice()->UpdateSurface(offscreenSurface, NULL, destinationSurface, NULL);
			offscreenSurface->Release();
		}
	}


	void DirectX9Texture::setGrid(Grid<Color>* a_pGrid)
	{
		HRESULT hr=S_OK;
		Grid<Color>* pGrid=NULL;
		D3DSURFACE_DESC desc;
		m_pTexture->GetLevelDesc(0, &desc);
		//TODO: get texture format from color
		if(D3DFMT_A32B32G32R32F==desc.Format)
		{
			IDirect3DSurface9* offscreenSurface;
			hr = m_pRenderer->getDevice()->CreateOffscreenPlainSurface( desc.Width, desc.Height, desc.Format, D3DPOOL_SYSTEMMEM, &offscreenSurface, NULL );
			if( FAILED(hr) )
				return;

			IDirect3DSurface9* destinationSurface;
			m_pTexture->GetSurfaceLevel(0, &destinationSurface);

			m_iWidth=desc.Width;
			m_iHeight=desc.Height;

			D3DLOCKED_RECT lockedRect;
			HRESULT hr=S_OK;
			if(!FAILED(offscreenSurface->LockRect(&lockedRect, NULL, D3DLOCK_DISCARD)))
			{
				for(unsigned int y=0; y<m_iHeight; y++)
				{
					void* dst=((char*)lockedRect.pBits)+(y*lockedRect.Pitch);
					memcpy(dst, a_pGrid->getRawRowData(y), a_pGrid->getRawRowDataByteCount());
				}
				offscreenSurface->UnlockRect();
			}
			m_pRenderer->getDevice()->UpdateSurface(offscreenSurface, NULL, destinationSurface, NULL);
			offscreenSurface->Release();
		}
	}

	unsigned int DirectX9Texture::getWidth(void)
	{
		//D3DSURFACE_DESC desc;
		//m_pTexture->GetLevelDesc(0, &desc);
		//return desc.Width;
		return m_iWidth;
	}

	unsigned int DirectX9Texture::getHeight(void)
	{
		//D3DSURFACE_DESC desc;
		//m_pTexture->GetLevelDesc(0, &desc);
		//return desc.Height;
		return m_iHeight;
	}
}