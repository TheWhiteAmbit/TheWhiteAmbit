#pragma once
#include "IMementoMementable.h"
namespace TheWhiteAmbit {
	class IOriginatorMementable
	{
	public:
		virtual IMementoMementable* saveToMemento(void)=0;
		virtual void restoreFromMemento(IMementoMementable* memento)=0;
		virtual ~IOriginatorMementable(void) {}
	};
}