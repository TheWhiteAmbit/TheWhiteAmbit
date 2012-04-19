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




#include "TransformVisitor.h"
namespace TheWhiteAmbit {
	TransformVisitor::TransformVisitor(void)
	{
		D3DXMatrixIdentity(&this->m_MartixWorld);
		D3DXMatrixIdentity(&this->m_MartixView);
		D3DXMatrixIdentity(&this->m_MartixProjection);
	}

	TransformVisitor::~TransformVisitor(void)
	{
	}

	void TransformVisitor::visitEnter(ITransformable* a_pSceneNode)
	{
		this->m_MartixWorld=a_pSceneNode->transformEnterWorld(this->m_MartixWorld);
		this->m_MartixView=a_pSceneNode->transformEnterView(this->m_MartixView);
		this->m_MartixProjection=a_pSceneNode->transformEnterProjection(this->m_MartixProjection);
	}

	void TransformVisitor::visitLeave(ITransformable* a_pSceneNode)
	{
		this->m_MartixWorld=a_pSceneNode->transformLeaveWorld(this->m_MartixWorld);
		this->m_MartixView=a_pSceneNode->transformLeaveView(this->m_MartixView);
		this->m_MartixProjection=a_pSceneNode->transformLeaveProjection(this->m_MartixProjection);	
	}
}
