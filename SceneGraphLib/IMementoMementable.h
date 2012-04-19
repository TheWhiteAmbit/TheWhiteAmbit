#pragma once
namespace TheWhiteAmbit {
	class IMementoMementable
	{
	public:
		virtual void* getSavedState()=0;
		virtual ~IMementoMementable(void) {}
	};
}