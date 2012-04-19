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

#include "../SceneGraphLib/Node.h"
#include "CameraAsset.h"
#include "PickIntersection.h"

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		public ref class BaseNode
		{
		protected:
			Node*	m_pNode;
			//TODO: find a way to handle structure in unmanaged scenegraph
			//but preserve/restore information in managed structure
			BaseNode^ parent;
			BaseNode^ child;
			BaseNode^ next;
		internal:		
			TheWhiteAmbit::Node* GetUnmanagedNode(void);
		public:
			void MakeParent(BaseNode^ a_pBaseNode);
			void SetChild(BaseNode^ a_pBaseNode);
			void SetNext(BaseNode^ a_pBaseNode);

			BaseNode^ GetParent(void);
			BaseNode^ GetChild(void);
			BaseNode^ GetNext(void);

			BaseNode();

			//TODO: move functions to renderer?
			void ApplyCamera(CameraAsset^ a_pCamera);
			void Memento(void);
			PickIntersection^ TestIntersection(System::IntPtr^ hWND);
			PickIntersection^ TestIntersection(double x, double y);
			void WndProc(System::Windows::Forms::Message% m);
		};
	}
}