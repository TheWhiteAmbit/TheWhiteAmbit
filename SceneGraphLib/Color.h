#pragma once
#include "../DirectXUTLib/DXUT.h"

namespace TheWhiteAmbit {
	class Color : public D3DXVECTOR4
	{
	public:
		Color(void);
		Color(double r, double g, double b, double a);
		virtual ~Color(void);
	};
}