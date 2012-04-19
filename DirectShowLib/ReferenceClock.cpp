
#include "ReferenceClock.h"
namespace TheWhiteAmbit {
	CReferenceClock::CReferenceClock( LPUNKNOWN pUnk, HRESULT *phr) 
		:CBaseReferenceClock(NAME("TheWhiteAmbit Clock"), pUnk, phr, 0)
	{
		*phr = S_OK;
		m_bPerfTimer = (TRUE==QueryPerformanceFrequency((LARGE_INTEGER *)&m_iTicksPerSecond));		
		if((!m_bPerfTimer)||(!m_iTicksPerSecond)||(m_iTicksPerSecond<=1000))
		{
			m_iTicksPerSecond = 1000;
			m_bPerfTimer = false;
		}
		m_iTicksNowLast=0;
		if(m_bPerfTimer)
		{
			QueryPerformanceCounter((LARGE_INTEGER *)&m_iTicksNowLast);
		}
		else
		{
			m_iTicksNowLast=(__int64)timeGetTime();
		}
		m_iTicksNow=0;
		m_fTimeFactor=1.0;
		m_fTimeAccumulated=0.0;
	}

	CReferenceClock::~CReferenceClock()
	{
	}

	REFERENCE_TIME CReferenceClock::GetPrivateTime(void)
	{
		m_iTicksNowLast=m_iTicksNow;
		if(m_bPerfTimer)
		{
			QueryPerformanceCounter((LARGE_INTEGER *)&m_iTicksNow);
		}
		else
		{
			m_iTicksNow=(__int64)timeGetTime();
		}
		m_fTimeAccumulated+=(((_abs64((__int64)(m_iTicksNow-m_iTicksNowLast)))*m_fTimeFactor)/(double)m_iTicksPerSecond)*10000000.0;
		return (__int64)m_fTimeAccumulated;
	}

	void CReferenceClock::setSpeed(double aSpeed)
	{
		m_fTimeFactor=aSpeed;
	}
}