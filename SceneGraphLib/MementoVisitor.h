#pragma once
#include "IOriginatorMementable.h"
namespace TheWhiteAmbit {
	class MementoVisitor
	{
	public:
		virtual void visit(IOriginatorMementable* a_pSceneNode);

		MementoVisitor(void);
		virtual ~MementoVisitor(void);
	};
}
