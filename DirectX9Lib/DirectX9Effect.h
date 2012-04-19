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



#pragma once
#include "DirectX9Renderer.h"
#include "DirectX9Texture.h"
#include "../SceneGraphLib/IEffect.h"

namespace TheWhiteAmbit {

	class DirectX9Effect : public IEffect
	{
		DirectX9Renderer* m_pRenderer;
		ID3DXEffect*    m_pEffect;
	public:
		DirectX9Effect(DirectX9Renderer* a_pRenderer, LPCWSTR a_sFilenameEffect);
		virtual ~DirectX9Effect(void);
		ID3DXEffect*    getEffect(void);
		void* getDirectX9Effect(void);
		void* getDirectX10Effect(void);
		void setValue(LPCWSTR name, double a_fValue);
		void setValue(LPCWSTR name, D3DXMATRIX* a_fValue);
		void setValue(LPCWSTR name, DirectX9Texture* a_pTexture);
	};
}