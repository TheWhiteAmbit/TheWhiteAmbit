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




#include "Trackball.h"
namespace TheWhiteAmbit {
	Trackball::Trackball(void)
	{
		D3DXMatrixIdentity(&m_TransformMatrixWorld);
		D3DXMatrixIdentity(&m_TransformMatrixView);
		D3DXMatrixIdentity(&m_TransformMatrixProjection);

		//m_pCamera=new CModelViewerCamera();
		m_pCamera=new CFirstPersonCamera();

		DXUTGetGlobalTimer()->Start();

		// Set the camera speed
		m_pCamera->SetScalers( 0.01f, 0.1f );

		m_pCamera->SetProjParams( D3DX_PI / 4, RESOLUTION_X/RESOLUTION_Y, 0.01f, 1000.0f );
		D3DXVECTOR3 vecEye( 0.0f, 0.0f, 0.0f );
		D3DXVECTOR3 vecAt ( 0.0f, 0.0f, 5.0f );
		m_pCamera->SetViewParams( &vecEye, &vecAt );

		m_pCamera->SetEnablePositionMovement(true);
		m_pCamera->SetResetCursorAfterMove(true);
		m_pCamera->SetNumberOfFramesToSmoothMouseData(1);
		// Don't constrain the camera to movement within the horizontal plane
		m_pCamera->SetEnableYAxisMovement( true );
	}

	Trackball::~Trackball(void)
	{
		if(m_pCamera)
			delete m_pCamera;
	}

	void Trackball::accept(WindowMessageVisitor* a_pWindowMessageVisitor)
	{
		a_pWindowMessageVisitor->visit(this);
	}

	LRESULT CALLBACK Trackball::windowProc( HWND a_hWindow, UINT a_Message, WPARAM a_WParam, LPARAM a_LParam )
	{
		LRESULT result=m_pCamera->HandleMessages(a_hWindow, a_Message, a_WParam, a_LParam);
		m_pCamera->FrameMove(0.1);
		//this->m_TransformMatrixWorld=*m_pCamera->GetWorldMatrix();
		this->m_TransformMatrixView=*m_pCamera->GetViewMatrix();
		//this->m_TransformMatrixProjection=*m_pCamera->GetProjMatrix();
		return result;
	}
}
