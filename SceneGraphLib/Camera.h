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
#include <iostream>
#include "../DirectXUTLib/DXUT.h"
#include "ITransformable.h"
#include "TransformVisitor.h"

namespace TheWhiteAmbit {
	class Camera:
		public ITransformable
	{
		D3DXMATRIX	m_TransformMatrixView;
		D3DXMATRIX	m_TransformMatrixProjection;
	public:
		Camera(void);
		~Camera(void);

		virtual D3DXMATRIX transformEnterWorld(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformLeaveWorld(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformEnterView(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformLeaveView(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformEnterProjection(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformLeaveProjection(D3DXMATRIX a_pTransform);

		virtual void acceptEnter(TransformVisitor* a_pTransformVisitor);

		//Calculate Projection
		void frustum(		double width,
			double height,
			double zNear,
			double zFar);
		void perspective( double fov, double aspect, double zNear, double zFar );
		void orthogonal( FLOAT width, FLOAT height, FLOAT zNear, FLOAT zFar);
		//Calculate View
		void lookAt( D3DXVECTOR3 vEyePt, D3DXVECTOR3 vLookatPt, D3DXVECTOR3 vUpVec);
		void moveInViewDir( D3DXVECTOR3 vOffset);
		void rotateView( D3DXVECTOR2 vOffset);
	};
}