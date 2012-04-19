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




#include "Node.h"
namespace TheWhiteAmbit {

	Node::Node(void)
	{
		m_pParent=NULL;
		m_pChild=NULL;
		m_pNext=NULL;
	}

	Node::~Node(void)
	{
	}

	void Node::acceptEnter(TransformVisitor* a_pNodeVisitor)
	{
		if(this->m_pChild)
			this->m_pChild->acceptEnter(a_pNodeVisitor);
		else this->acceptLeave(a_pNodeVisitor);
	}

	void Node::acceptLeave(TransformVisitor* a_pNodeVisitor)
	{
		if(this->m_pNext)
			this->m_pNext->acceptEnter(a_pNodeVisitor);
		else if(this->m_pParent)
			this->m_pParent->acceptLeave(a_pNodeVisitor);
	}

	void Node::accept(RenderVisitor* a_pNodeVisitor)
	{
		if(this->m_pChild)
			this->m_pChild->accept(a_pNodeVisitor);
		if(this->m_pNext)
			this->m_pNext->accept(a_pNodeVisitor);
	}

	void Node::accept(PickVisitor* a_pNodeVisitor)
	{
		if(this->m_pChild)
			this->m_pChild->accept(a_pNodeVisitor);
		if(this->m_pNext)
			this->m_pNext->accept(a_pNodeVisitor);
	}

	void Node::accept(RaytraceVisitor* a_pNodeVisitor)
	{
		if(this->m_pChild)
			this->m_pChild->accept(a_pNodeVisitor);
		if(this->m_pNext)
			this->m_pNext->accept(a_pNodeVisitor);
	}

	void Node::accept(MementoVisitor* a_pNodeVisitor)
	{
		if(this->m_pChild)
			this->m_pChild->accept(a_pNodeVisitor);
		if(this->m_pNext)
			this->m_pNext->accept(a_pNodeVisitor);
	}

	void Node::accept(WindowMessageVisitor* a_pNodeVisitor)
	{
		if(this->m_pChild)
			this->m_pChild->accept(a_pNodeVisitor);
		if(this->m_pNext)
			this->m_pNext->accept(a_pNodeVisitor);
	}

	void Node::makeParent(Node* a_pParent)
	{
		m_pParent=a_pParent;
		Node* child = m_pParent->getChild();
		if(!child){
			m_pParent->setChild(this);
			return;
		}
		while(child->getNext())
			child=child->getNext();
		child->setNext(this);
	}

	void Node::setNext(Node* a_pNext)
	{
		m_pNext=a_pNext;
	}

	void Node::setChild(Node* a_pChild)
	{
		m_pChild=a_pChild;
	}

	Node* Node::getParent(void)
	{
		return m_pParent;
	}

	Node* Node::getChild(void)
	{
		return m_pChild;
	}

	Node* Node::getNext(void)
	{
		return m_pNext;
	}
}