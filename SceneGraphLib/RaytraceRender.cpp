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



#include <omp.h>
#include "RaytraceRender.h"

namespace TheWhiteAmbit {

	RaytraceRender::RaytraceRender(unsigned int a_iWidth, unsigned int a_iHeight)
	{
		m_pRootNode=NULL;
		m_iTicks=0;
		this->m_pRenderGrid=new Grid<Color>(a_iWidth, a_iHeight);
	}

	RaytraceRender::~RaytraceRender(void)
	{
		delete this->m_pRenderGrid;
	}

	void RaytraceRender::setRootNode(Node* a_pNode)
	{
		m_pRootNode=a_pNode;
	}

	void RaytraceRender::present(void)
	{
		if(m_pRootNode)
		{
			unsigned int width=m_pRenderGrid->getWidth();
			unsigned int height=m_pRenderGrid->getHeight();

#pragma omp parallel for
			for(int y=0; y<(int)height; y++)
			{
				double normY=-(((2.0*y)/(double)height)-1.0);	//invert y
				for(int x=0; x<(int)width; x++)
				{
					double normX=((2.0*x)/(double)width)-1.0;
					PickVisitor pickNodeVisitor(D3DXVECTOR2((FLOAT)normX, (FLOAT)normY));
					this->m_pRootNode->accept(&pickNodeVisitor);
					Intersection intersection=pickNodeVisitor.getIntersection();

					Color pixel=Color(
						1.0f/intersection.fDist, 
						1.0f/intersection.fDist,
						1.0f/intersection.fDist, 
						.5);
					m_pRenderGrid->setPixel(x, y, pixel);
				}
			}
		}
	}

	Grid<Color>*	RaytraceRender::getGrid(void)
	{
		return this->m_pRenderGrid;
	}
}