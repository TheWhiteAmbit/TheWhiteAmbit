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



#include "Camera.h"

namespace TheWhiteAmbit {
	Camera::Camera(void)
	{
		//D3DXMatrixIdentity(&m_TransformMatrixWorld); //none
		D3DXMatrixIdentity(&m_TransformMatrixView);
		D3DXMatrixIdentity(&m_TransformMatrixProjection);
	}

	Camera::~Camera(void)
	{
	}


	D3DXMATRIX Camera::transformEnterWorld(D3DXMATRIX a_Transform)
	{
		return a_Transform;
	}

	D3DXMATRIX Camera::transformLeaveWorld(D3DXMATRIX a_Transform)
	{
		return a_Transform;
	}

	D3DXMATRIX Camera::transformEnterView(D3DXMATRIX a_Transform)
	{
		D3DXMATRIX result;
		D3DXMatrixMultiply(&result, &this->m_TransformMatrixView, &a_Transform);
		return result;
	}
	D3DXMATRIX Camera::transformLeaveView(D3DXMATRIX a_Transform)
	{
		return a_Transform;
	}

	D3DXMATRIX Camera::transformEnterProjection(D3DXMATRIX a_Transform)
	{
		D3DXMATRIX result;
		D3DXMatrixMultiply(&result, &this->m_TransformMatrixProjection, &a_Transform);
		return result;
	}

	D3DXMATRIX Camera::transformLeaveProjection(D3DXMATRIX a_Transform)
	{
		return a_Transform;
	}

	void Camera::acceptEnter(TransformVisitor* a_pTransformVisitor)
	{
		a_pTransformVisitor->visitEnter(this);
	}

	void Camera::perspective( double fov, double aspect, double zNear, double zFar )
	{
		const double pi = 3.1415926535897932384626433832795;  
		double fH;
		double fW;

		//fH = tan( (fovY / 2) / 180 * pi ) * zNear;

		if(aspect>1.0)
		{
			fH = tan( fov / 360 * pi ) * zNear;
			fW= fH * aspect;
		}
		else
		{
			fW = tan( fov / 360 * pi ) * zNear;
			fH= fW * 1.0/aspect;
		}

		frustum( 2.0*fW, 2.0*fH, zNear, zFar );
	}

	void Camera::frustum(	double width,
		double height,
		double zNear,
		double zFar)
	{
		double x=	(2.0*zNear)	/(width);
		double y=	(2.0*zNear)	/(height);
		double c;
		double d;

		if(zFar==0.0) //if farZ is set to 0.0 set far to INF
		{
			c=1.0;
			d=-2.0*zNear;
		}
		else 
		{
			c=(zFar)/(zFar-zNear);
			d=(zNear*zFar)/(zNear-zFar);
		}

		m_TransformMatrixProjection = D3DXMATRIX(	(float)x, 0, 0, 0,
			0, (float)y, 0, 0,
			0, 0, (float)c, 1,
			0, 0, (float)d, 0);

	}

	void Camera::orthogonal( FLOAT width, FLOAT height, FLOAT zNear, FLOAT zFar)
	{
		// Get D3DX to fill in the matrix values
		D3DXMatrixOrthoLH( &m_TransformMatrixProjection, width, height, zNear, zFar);
	}

	void Camera::lookAt( D3DXVECTOR3 vEyePt, D3DXVECTOR3 vLookatPt, D3DXVECTOR3 vUpVec)
	{
		//// Initialise our vectors
		//D3DXVECTOR3 vEyePt( x, y, z );
		//D3DXVECTOR3 vLookatPt( 0.0f, 0.0f, 0.0f );
		//D3DXVECTOR3 vUpVec( 0.0f, 1.0f, 0.0f );

		// Get D3DX to fill in the matrix values
		D3DXMatrixLookAtLH( &m_TransformMatrixView, &vEyePt, &vLookatPt, &vUpVec );

		//Implemented as follows
		//zaxis = normal(At - Eye)
		//xaxis = normal(cross(Up, zaxis))
		//yaxis = cross(zaxis, xaxis)

		//xaxis.x           yaxis.x           zaxis.x          0
		//xaxis.y           yaxis.y           zaxis.y          0
		//xaxis.z           yaxis.z           zaxis.z          0
		//-dot(xaxis, eye)  -dot(yaxis, eye)  -dot(zaxis, eye)  1
	}

	void Camera::moveInViewDir( D3DXVECTOR3 vOffset)
	{

		D3DXVECTOR3 vOffsetViewDirection;
		D3DXVec3TransformNormal(&vOffsetViewDirection, &vOffset, &this->m_TransformMatrixView);
		D3DXMATRIX matDiff;
		D3DXMatrixAffineTransformation(&matDiff, 1.0, NULL, NULL, &vOffsetViewDirection);
		//TODO: correct move view dir function
		D3DXMatrixMultiply(&this->m_TransformMatrixView, &matDiff, &this->m_TransformMatrixView);
	}

	void Camera::rotateView( D3DXVECTOR2 vOffset)
	{
		//TODO: implement rotation
	}
}