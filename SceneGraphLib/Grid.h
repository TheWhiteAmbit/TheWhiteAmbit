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
namespace TheWhiteAmbit {

	template <typename T>
	class Grid
	{
		T* m_pData;
		unsigned int m_iWidth;
		unsigned int m_iHeight;
	public:
		Grid(unsigned int width, unsigned int height);
		~Grid(void);

		T& getPixel(unsigned int a_XPos, unsigned int a_YPos);
		void setPixel(unsigned int a_XPos, unsigned int a_YPos, T& a_Pixel);
		unsigned int getBytesPerPixel(void);
		unsigned int getWidth(void);
		unsigned int getHeight(void);
		T* getRawData(void);
		T* getRawRowData(unsigned int a_iRow);
		unsigned int getRawDataByteCount(void);
		unsigned int getRawRowDataByteCount(void);
	};


	//since this is a template class the function definitions have to be in the header

	template <typename T>
	Grid<T>::Grid(unsigned int width, unsigned int height)
	{
		m_iWidth=width;
		m_iHeight=height;
		this->m_pData=new T[m_iWidth*m_iHeight];
	}

	template <typename T>
	Grid<T>::~Grid(void)
	{
		delete[] this->m_pData;
	}

	template <typename T>
	T& Grid<T>::getPixel(unsigned int a_XPos, unsigned int a_YPos)
	{
		a_XPos%=m_iWidth;
		a_YPos%=m_iHeight;
		return m_pData[a_XPos+a_YPos*m_iWidth];
	}

	template <typename T>
	void Grid<T>::setPixel(unsigned int a_XPos, unsigned int a_YPos, T& a_Pixel)
	{
		a_XPos%=m_iWidth;
		a_YPos%=m_iHeight;
		void* dst=(void*)m_pData[a_XPos+a_YPos*m_iWidth];
		void* src=(void*)&a_Pixel;
		memcpy(dst, src, sizeof(T));
	}

	template <typename T>
	T* Grid<T>::getRawData(void)
	{
		return m_pData;
	}

	template <typename T>
	T* Grid<T>::getRawRowData(unsigned int a_iRow)
	{
		a_iRow%=m_iHeight;
		return &m_pData[a_iRow*m_iWidth];
	}

	template <typename T>
	unsigned int Grid<T>::getRawDataByteCount(void)
	{
		return m_iWidth*m_iHeight*getBytesPerPixel();
	}

	template <typename T>
	unsigned int Grid<T>::getRawRowDataByteCount(void)
	{
		return m_iWidth*getBytesPerPixel();
	}

	template <typename T>
	unsigned int Grid<T>::getBytesPerPixel(void)
	{
		return sizeof(T);
	}

	template <typename T>
	unsigned int Grid<T>::getWidth(void)
	{
		return m_iWidth;
	}

	template <typename T>
	unsigned int Grid<T>::getHeight(void)
	{
		return m_iHeight;
	}
}

