
#include "MementoVisitor.h"
namespace TheWhiteAmbit {
	MementoVisitor::MementoVisitor(void)
	{
	}

	MementoVisitor::~MementoVisitor(void)
	{
	}

	void MementoVisitor::visit(IOriginatorMementable* a_pSceneNode) {
		//TODO: remove this "autistic" unspecific behavior - works for DirectX9DrawMock
		a_pSceneNode->restoreFromMemento(a_pSceneNode->saveToMemento());
	}
}