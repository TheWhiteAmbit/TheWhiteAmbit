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
#include "../SceneGraphLib/Camera.h"
namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		public ref class CameraAsset
		{
			TheWhiteAmbit::Camera* m_pCamera;
		internal:
			TheWhiteAmbit::Camera* GetUnmanagedAsset(void);
		public:		
			CameraAsset();
			//Calculate Projection
			void Frustum(
				double width,
				double height,
				double nearZ,
				double farZ);
			void Perspective( double fov, double aspect, double zNear, double zFar );
			//Calculate View
			void LookAt( double vEyePtX,	double vEyePtY,		double vEyePtZ, 
				double vLookatPtX, double vLookatPtY,	double vLookatPtZ,
				double vUpVecX,	double vUpVecY,		double vUpVecZ);
			void MoveInViewDir(double x, double y, double z);
			void Orthogonal( double Width, double Height, double zNear, double zFar);
		};
	}
}