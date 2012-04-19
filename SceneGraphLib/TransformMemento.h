#pragma once
#include "ITransformable.h"
#include "IMementoMementable.h"
namespace TheWhiteAmbit {
	class TransformMemento :
		public ITransformable, IMementoMementable
	{
	protected:
		D3DXMATRIX	m_TransformMatrixWorld;
		D3DXMATRIX	m_BackupMatrixWorld;
		D3DXMATRIX	m_TransformMatrixView;
		D3DXMATRIX	m_BackupMatrixView;
		D3DXMATRIX	m_TransformMatrixProjection;
		D3DXMATRIX	m_BackupMatrixProjection;
		ITransformable*	m_BackupTransform;
	public:
		void setWorld(D3DXMATRIX a_pTransform);
		void setView(D3DXMATRIX a_pTransform);
		void setProjection(D3DXMATRIX a_pTransform);

		virtual D3DXMATRIX transformEnterWorld(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformLeaveWorld(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformEnterView(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformLeaveView(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformEnterProjection(D3DXMATRIX a_pTransform);
		virtual D3DXMATRIX transformLeaveProjection(D3DXMATRIX a_pTransform);

		virtual void* getSavedState();

		TransformMemento(void);
		virtual ~TransformMemento();
	};
}
