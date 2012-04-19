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
		Node* BaseNode::GetUnmanagedNode(void)
		{
			return this->m_pNode;
		}

		void BaseNode::MakeParent(BaseNode^ a_pParentNode)
		{
			this->m_pNode->makeParent(a_pParentNode->m_pNode);

			this->parent=a_pParentNode;
			BaseNode^ child = this->parent->GetChild();
			if(!child){
				parent->SetChild(this);
				return;
			}
			while(child->GetNext())
				child=child->GetNext();
			child->SetNext(this);
		}

		void BaseNode::SetChild(BaseNode^ a_pChildNode)
		{
			this->m_pNode->setChild(a_pChildNode->m_pNode);
			this->child=a_pChildNode;
		}

		void BaseNode::SetNext(BaseNode^ a_pNextNode)
		{
			this->m_pNode->setNext(a_pNextNode->m_pNode);
			this->next=a_pNextNode;
		}

		BaseNode^ BaseNode::GetParent(void)
		{
			return this->parent;
		}

		BaseNode^ BaseNode::GetChild(void)
		{
			return this->child;
		}

		BaseNode^ BaseNode::GetNext(void)
		{
			return this->next;
		}

		BaseNode::BaseNode()
		{		
			this->parent=nullptr;
			this->child=nullptr;
			this->next=nullptr;

			this->m_pNode=new Node();
		}

		void BaseNode::ApplyCamera(CameraAsset^ a_pCamera)
		{
			Camera* camera=(Camera*)a_pCamera->GetUnmanagedAsset();
			TransformVisitor transformNodeVisitor;
			camera->acceptEnter(&transformNodeVisitor);
			this->m_pNode->acceptEnter(&transformNodeVisitor);
		}

		void BaseNode::Memento()
		{
			MementoVisitor mementoVisitor;
			this->m_pNode->accept(&mementoVisitor);
		}

		PickIntersection^ BaseNode::TestIntersection(System::IntPtr^ hWND)
		{
			PickVisitor pickNodeVisitor((HWND)hWND->ToPointer());
			this->m_pNode->accept(&pickNodeVisitor);

			Intersection intersection=pickNodeVisitor.getIntersection();

			PickIntersection^ managedIntersection=gcnew PickIntersection();
			managedIntersection->face=intersection.dwFace;
			managedIntersection->barycentric1=intersection.fBary1;
			managedIntersection->barycentric2=intersection.fBary2;
			managedIntersection->distance=intersection.fDist;
			managedIntersection->texture_u=intersection.tu;
			managedIntersection->texture_v=intersection.tv;
			managedIntersection->backface=intersection.bBackface!=0;

			return managedIntersection;
		}

		void BaseNode::WndProc(System::Windows::Forms::Message% m)
		{
			WindowMessageVisitor messageVisitor((HWND)m.HWnd.ToPointer(), m.Msg, (WPARAM)m.WParam.ToPointer(), (LPARAM)m.LParam.ToPointer());
			//WindowMessageVisitor messageVisitor((HWND)m.HWnd.ToPointer(), m.Msg, (WPARAM)m.WParam.ToPointer(), (LPARAM)m.LParam.ToPointer());
			this->m_pNode->accept(&messageVisitor);
		}

		PickIntersection^ BaseNode::TestIntersection(double x, double y)
		{
			PickVisitor pickNodeVisitor(D3DXVECTOR2((FLOAT)x,(FLOAT)y));
			this->m_pNode->accept(&pickNodeVisitor);

			Intersection intersection=pickNodeVisitor.getIntersection();

			PickIntersection^ managedIntersection=gcnew PickIntersection();
			managedIntersection->face=intersection.dwFace;
			managedIntersection->barycentric1=intersection.fBary1;
			managedIntersection->barycentric2=intersection.fBary2;
			managedIntersection->distance=intersection.fDist;
			managedIntersection->texture_u=intersection.tu;
			managedIntersection->texture_v=intersection.tv;
			managedIntersection->backface=intersection.bBackface!=0;

			return managedIntersection;
		}
	}
}