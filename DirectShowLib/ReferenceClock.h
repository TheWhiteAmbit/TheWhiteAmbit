

#include <streams.h>
namespace TheWhiteAmbit {
	class CReferenceClock : public CBaseReferenceClock
	{
		__int64	m_iTicksPerSecond;
		__int64	m_iTicksNow;
		__int64	m_iTicksNowLast;
		double	m_fTimeAccumulated;
		double  m_fTimeFactor;
		bool	m_bPerfTimer;
	public:
		void setSpeed(double);
		virtual REFERENCE_TIME GetPrivateTime(void);
		CReferenceClock(LPUNKNOWN pUnk,HRESULT *phr);
		virtual ~CReferenceClock();
	};
}