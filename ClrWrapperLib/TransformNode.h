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

#include "BaseNode.h"

using namespace System;

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		public ref class TransformNode :
			public BaseNode
			{
			public:
				TransformNode();
				void SetMatrixWorld(
					double m11, double m12, double m13, double m14, 
					double m21, double m22, double m23, double m24, 
					double m31, double m32, double m33, double m34, 
					double m41, double m42, double m43, double m44);
			};
	}
}