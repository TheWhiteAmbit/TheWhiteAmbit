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



#include "stdafx.h"

#include "CameraAsset.h"

using namespace System;

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		CameraAsset::CameraAsset()
		{
			this->m_pCamera=new Camera();
		}

		Camera* CameraAsset::GetUnmanagedAsset(void)
		{
			return this->m_pCamera;
		}

		void CameraAsset::Frustum(		double width,
			double height,
			double nearZ,
			double farZ)
		{
			this->m_pCamera->frustum(width, height, nearZ, farZ);
		}
		void CameraAsset::Perspective( double fov, double aspect, double zNear, double zFar )
		{
			this->m_pCamera->perspective( fov, aspect, zNear, zFar );
		}
		//Calculate View
		void CameraAsset::LookAt( double vEyePtX,	double vEyePtY,		double vEyePtZ, 
			double vLookatPtX, double vLookatPtY,	double vLookatPtZ,
			double vUpVecX,	double vUpVecY,		double vUpVecZ)
		{
			this->m_pCamera->lookAt( 
				D3DXVECTOR3( (float)vEyePtX, (float)vEyePtY, (float)vEyePtZ), 
				D3DXVECTOR3( (float)vLookatPtX, (float)vLookatPtY, (float)vLookatPtZ),
				D3DXVECTOR3( (float)vUpVecX, (float)vUpVecY, (float)vUpVecZ));
		}

		void CameraAsset::Orthogonal( double Width, double Height, double zNear,		double zFar)
		{
			this->m_pCamera->orthogonal( Width, Height, zNear, zFar);
		}

		void CameraAsset::MoveInViewDir(double x, double y, double z)
		{
			this->m_pCamera->moveInViewDir(D3DXVECTOR3((FLOAT)x,(FLOAT)y,(FLOAT)z));
		}
	}
}