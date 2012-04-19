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
#include "IPickable.h"
#include "Intersection.h"
#include "Ray.h"
#include "Sphere.h"
#include "Face.h"

#include <iostream>
#include "../DirectXUTLib/DXUT.h"

#define MAX_INTERSECTIONS 1

namespace TheWhiteAmbit {

	class PickVisitor
	{
	protected:
		Intersection	m_Intersection; // Intersection info
		D3DXVECTOR2		m_ScreenPick;
		Ray*			m_pWorldViewRay;

		//Pick normalized ScreenSpace coordinate
		D3DXVECTOR2 PickScreen(HWND hWND);

		//Pick WorldView direction
		D3DXVECTOR4 PickDirection(
			D3DXVECTOR2 a_ScreenPick, 
			D3DXMATRIX matProj);

		//Pick ObjectSpace ray
		Ray PickInvMatrix(Ray ray, 
			D3DXMATRIX a_Matrix);

	public:

		PickVisitor(HWND a_WindowHandle);
		PickVisitor(D3DXVECTOR2 a_ClickPoint);
		PickVisitor(Ray a_Ray);
		virtual ~PickVisitor(void);
		virtual void visit(IPickable* a_pSceneNode);
		virtual Intersection getIntersection(void);
	};
}
