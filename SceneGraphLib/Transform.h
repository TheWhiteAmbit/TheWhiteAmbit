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
#include "Node.h"
#include "ITransformable.h"
namespace TheWhiteAmbit {
	class Transform :
		public Node, ITransformable
	{
	protected:
		D3DXMATRIX	m_TransformMatrixWorld;
		D3DXMATRIX	m_BackupMatrixWorld;
		D3DXMATRIX	m_TransformMatrixView;
		D3DXMATRIX	m_BackupMatrixView;
		D3DXMATRIX	m_TransformMatrixProjection;
		D3DXMATRIX	m_BackupMatrixProjection;
		ITransformable*	m_BackupTransform;
	public:
		void setWorld(D3DXMATRIX a_pTransform);
		void setView(D3DXMATRIX a_pTransform);
		void setProjection(D3DXMATRIX a_pTransform);

		Transform(void);
		virtual ~Transform(void);

		virtual D3DXMATRIX transformEnterWorld(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformLeaveWorld(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformEnterView(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformLeaveView(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformEnterProjection(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformLeaveProjection(D3DXMATRIX a_pTransform);
		virtual void acceptEnter(TransformVisitor* a_pNodeVisitor);
		virtual void acceptLeave(TransformVisitor* a_pNodeVisitor);
	};
}