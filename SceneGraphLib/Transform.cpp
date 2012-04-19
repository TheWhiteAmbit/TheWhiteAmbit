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




#include "Transform.h"

namespace TheWhiteAmbit {
	Transform::Transform()
	{
		D3DXMatrixIdentity(&m_TransformMatrixWorld);
		D3DXMatrixIdentity(&m_BackupMatrixWorld);
		D3DXMatrixIdentity(&m_TransformMatrixView);
		D3DXMatrixIdentity(&m_BackupMatrixView);
		D3DXMatrixIdentity(&m_TransformMatrixProjection);
		D3DXMatrixIdentity(&m_BackupMatrixProjection);
	}

	Transform::~Transform(void)
	{
	}

	void Transform::setWorld(D3DXMATRIX a_pTransform)
	{
		m_TransformMatrixWorld=a_pTransform;
	}
	void Transform::setView(D3DXMATRIX a_pTransform)
	{
		m_TransformMatrixView=a_pTransform;
	}
	void Transform::setProjection(D3DXMATRIX a_pTransform)
	{
		m_TransformMatrixProjection=a_pTransform;
	}


	D3DXMATRIX Transform::transformEnterWorld(D3DXMATRIX a_Transform)
	{
		D3DXMATRIX result;
		m_BackupMatrixWorld=a_Transform;
		D3DXMatrixMultiply(&result, &this->m_TransformMatrixWorld, &a_Transform);
		return result;
	}

	D3DXMATRIX Transform::transformLeaveWorld(D3DXMATRIX a_Transform)
	{
		return m_BackupMatrixWorld;
	}

	D3DXMATRIX Transform::transformEnterView(D3DXMATRIX a_Transform)
	{
		D3DXMATRIX result;
		m_BackupMatrixView=a_Transform;
		D3DXMatrixMultiply(&result, &this->m_TransformMatrixView, &a_Transform);
		return result;
	}
	D3DXMATRIX Transform::transformLeaveView(D3DXMATRIX a_Transform)
	{
		return m_BackupMatrixView;
	}

	D3DXMATRIX Transform::transformEnterProjection(D3DXMATRIX a_Transform)
	{
		D3DXMATRIX result;
		m_BackupMatrixProjection=a_Transform;
		D3DXMatrixMultiply(&result, &this->m_TransformMatrixProjection, &a_Transform);
		return result;
	}

	D3DXMATRIX Transform::transformLeaveProjection(D3DXMATRIX a_Transform)
	{
		return m_BackupMatrixProjection;
	}

	void Transform::acceptEnter(TransformVisitor* a_pTransformVisitor)
	{
		a_pTransformVisitor->visitEnter(this);
		Node::acceptEnter(a_pTransformVisitor);
	}

	void Transform::acceptLeave(TransformVisitor* a_pTransformVisitor)
	{
		a_pTransformVisitor->visitLeave(this);
		Node::acceptLeave(a_pTransformVisitor);
	}
}