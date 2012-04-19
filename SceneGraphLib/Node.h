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
#include "RenderVisitor.h"
#include "TransformVisitor.h"
#include "PickVisitor.h"
#include "RaytraceVisitor.h"
#include "MementoVisitor.h"
#include "WindowMessageVisitor.h"
namespace TheWhiteAmbit {
	class Node
	{
	protected:
		Node* m_pParent;
		Node* m_pNext;
		Node* m_pChild;
	public:
		virtual void acceptEnter(TransformVisitor* a_pNodeVisitor);
		virtual void acceptLeave(TransformVisitor* a_pNodeVisitor);
		virtual void accept(RenderVisitor* a_pNodeVisitor);	
		virtual void accept(PickVisitor* a_pNodeVisitor);
		virtual void accept(RaytraceVisitor* a_pNodeVisitor);
		virtual void accept(MementoVisitor* a_pNodeVisitor);
		virtual void accept(WindowMessageVisitor* a_pWindowMessageVisitor);

		virtual Node* getParent(void);
		virtual Node* getNext(void);
		virtual Node* getChild(void);

		virtual void makeParent(Node* a_pParent);
		virtual void setNext(Node* a_pNext);
		virtual void setChild(Node* a_pChild);

		Node(void);
		virtual ~Node(void);
	};
}
