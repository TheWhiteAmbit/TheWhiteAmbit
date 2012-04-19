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

#include "../SceneGraphLib/Transform.h"
#include "TransformNode.h"

using namespace System;

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		TransformNode::TransformNode()
		{
			this->m_pNode=new Transform();
		}

		void TransformNode::SetMatrixWorld(
			double m11, double m12, double m13, double m14, 
			double m21, double m22, double m23, double m24, 
			double m31, double m32, double m33, double m34, 
			double m41, double m42, double m43, double m44)
		{
			D3DMATRIX m;
			m._11=(float)m11; m._12=(float)m12; m._13=(float)m13; m._14=(float)m14;
			m._21=(float)m21; m._22=(float)m22; m._23=(float)m23; m._24=(float)m24;
			m._31=(float)m31; m._32=(float)m32; m._33=(float)m33; m._34=(float)m34;
			m._41=(float)m41; m._42=(float)m42; m._43=(float)m43; m._44=(float)m44;
			Transform* transform=(Transform*)this->m_pNode;
			//transform->transformEnterWorld(m);
			transform->setWorld(m);
		}
	}
}