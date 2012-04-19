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

#include "TransformMemento.h"
namespace TheWhiteAmbit {
	TransformMemento::TransformMemento() {
		D3DXMatrixIdentity(&m_TransformMatrixWorld);
		D3DXMatrixIdentity(&m_BackupMatrixWorld);
		D3DXMatrixIdentity(&m_TransformMatrixView);
		D3DXMatrixIdentity(&m_BackupMatrixView);
		D3DXMatrixIdentity(&m_TransformMatrixProjection);
		D3DXMatrixIdentity(&m_BackupMatrixProjection);
	}

	TransformMemento::~TransformMemento(void) {
	}

	void TransformMemento::setWorld(D3DXMATRIX a_pTransform) {
		m_TransformMatrixWorld=a_pTransform;
	}

	void TransformMemento::setView(D3DXMATRIX a_pTransform) {
		m_TransformMatrixView=a_pTransform;
	}

	void TransformMemento::setProjection(D3DXMATRIX a_pTransform) {
		m_TransformMatrixProjection=a_pTransform;
	}


	D3DXMATRIX TransformMemento::transformEnterWorld(D3DXMATRIX a_Transform) {
		m_BackupMatrixWorld=a_Transform;
		return m_TransformMatrixWorld;
	}

	D3DXMATRIX TransformMemento::transformLeaveWorld(D3DXMATRIX a_Transform) {
		return m_BackupMatrixWorld;
	}

	D3DXMATRIX TransformMemento::transformEnterView(D3DXMATRIX a_Transform) {
		m_BackupMatrixView=a_Transform;
		return m_TransformMatrixView;
	}
	D3DXMATRIX TransformMemento::transformLeaveView(D3DXMATRIX a_Transform) {
		return m_BackupMatrixView;
	}

	D3DXMATRIX TransformMemento::transformEnterProjection(D3DXMATRIX a_Transform) {
		m_BackupMatrixProjection=a_Transform;
		return m_TransformMatrixProjection;
	}

	D3DXMATRIX TransformMemento::transformLeaveProjection(D3DXMATRIX a_Transform) {
		return m_BackupMatrixProjection;
	}

	void* TransformMemento::getSavedState() {
		return this; //TODO: look if this is good and used at all?
	}
}