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



// ClrWrapperLib.h
#pragma once
#include "PickIntersection.h"
#include "RaytraceRenderAsset.h"

#include "CameraAsset.h"
#include "EffectAsset.h"
#include "SdkMeshAsset.h"
#include "ObjMeshAsset.h"
#include "TargetRenderAsset.h"
#include "ViewRenderAsset.h"
#include "TextureAsset.h"
#include "CudaRaytraceRenderAsset.h"
#include "CudaVideoRenderAsset.h"
#include "CudaFileRenderAsset.h"
#include "DirectShowFileRenderAsset.h"
#include "MediaFoundationFileRenderAsset.h"

#include "BaseNode.h"
#include "DrawMockNode.h"
#include "MaterialNode.h"
#include "TextureLayerNode.h"
#include "ClearBufferNode.h"
#include "TransformNode.h"
#include "TrackballNode.h"

//Potentially used librarys
//user32.lib gdi32.lib shell32.lib

using namespace System;
using namespace System::Drawing;

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		public ref class SceneGraphFactory
		{
			DirectX9Renderer*	m_pRenderer;
		public:
			// Nodes
			DrawMockNode^			CreateDrawMock(void);
			BaseNode^				CreateBaseNode(void);
			ClearBufferNode^		CreateClearBuffer(void);
			TransformNode^			CreateTransform(void);
			TrackballNode^			CreateTrackball(void);
			MaterialNode^			CreateMaterial(void);
			TextureLayerNode^		CreateTextureLayer(void);
			TextureLayerNode^		CreateTextureLayer(TargetRenderAsset^ a_pTarget);
			TextureLayerNode^		CreateTextureLayer(RaytraceRenderAsset^ a_pTarget);
			TextureLayerNode^		CreateTextureLayer(CudaRaytraceRenderAsset^ a_pTarget);
			TextureLayerNode^		CreateTextureLayer(CudaVideoRenderAsset^ a_pTarget);
			TextureLayerNode^		CreateTextureLayer(TextureAsset^ a_pTexture, TextureAsset^ a_pTexture1);
			TextureAsset^			CreateTextureAsset(TargetRenderAsset^ a_pTarget);			
			TextureAsset^			CreateTextureAsset(TextureLayerNode^ a_pTextureLayerNode);
			TextureAsset^			CreateTextureAsset(CudaVideoRenderAsset^ a_pCudaVideoRenderAsset);
			TextureAsset^			CreateTextureAsset(Bitmap^ a_pBitmap);

			Bitmap^					CreateBitmap(TextureAsset^ a_pTexture);
			GridAsset^	            CreateGrid(TextureAsset^ a_pTexture);
			Bitmap^                 CreateBitmap(GridAsset^ a_pGrid);

			// Loaded Assets
			EffectAsset^			CreateEffect(System::String^ a_sFilename);
			SdkMeshAsset^			CreateSdkMesh(System::String^ a_sFilename);
			ObjMeshAsset^			CreateObjMesh(System::String^ a_sFilename);

			// Generated Assets
			CameraAsset^			CreateCamera(void);
			ViewRenderAsset^		CreateView(void);
			TargetRenderAsset^		CreateRenderTargetAsset(void);
			TargetRenderAsset^		CreateRenderTargetA8R8G8B8Asset(void);
			RaytraceRenderAsset^	CreateRaytracer(void);

			// Generate Cuda Assets
			CudaRaytraceRenderAsset^	CreateCudaRaytraceRenderAsset(void);
			CudaVideoRenderAsset^	CreateCudaVideoRenderAsset(System::String^ a_sFilename);
			CudaFileRenderAsset^	CreateCudaFileRenderAsset(System::String^ a_sFilename);

			// Generate DirectShow (MediaFoundation) Assets
			DirectShowFileRenderAsset^	CreateDirectShowFileRenderAsset(System::String^ a_sFilename);
			MediaFoundationFileRenderAsset^	CreateMediaFoundationFileRenderAsset(System::String^ a_sFilename, unsigned int width, unsigned int height);

			SceneGraphFactory(System::IntPtr^ hWND);
			SceneGraphFactory();
			~SceneGraphFactory(void);
		};
	}
}
