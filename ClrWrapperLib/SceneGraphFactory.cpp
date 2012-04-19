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



// This is the main DLL file.

#include "stdafx.h"

#include "SceneGraphFactory.h"

using namespace System;

namespace TheWhiteAmbit {
	namespace ClrWrapperLib {
		SceneGraphFactory::SceneGraphFactory(System::IntPtr^ hWND)
		{
			m_pRenderer=new DirectX9Renderer((HWND)hWND->ToPointer());
		}

		SceneGraphFactory::SceneGraphFactory()
		{
			// set up the structure used to create the D3DDevice
			D3DPRESENT_PARAMETERS d3dpp;
			ZeroMemory(&d3dpp, sizeof(d3dpp));
			d3dpp.Windowed = TRUE;
			//d3dpp.BackBufferWidth = 1;
			//d3dpp.BackBufferHeight = 1;
			d3dpp.BackBufferWidth = (UINT)RESOLUTION_X;
			d3dpp.BackBufferHeight = (UINT)RESOLUTION_Y;
			d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
			d3dpp.EnableAutoDepthStencil = TRUE;
			d3dpp.AutoDepthStencilFormat = D3DFMT_D24S8;
			d3dpp.BackBufferFormat = D3DFMT_UNKNOWN;
			//d3dpp.BackBufferFormat = D3DFMT_A8R8G8B8;

			// create the device's window
			WNDCLASSEX wndClassEx = { sizeof(WNDCLASSEX), CS_CLASSDC, DefWindowProc, 0L, 0L,

				GetModuleHandle(NULL), NULL, NULL, NULL, NULL, L"HiddenD3DtoWPFWindow", NULL };
			RegisterClassEx(&wndClassEx);

			HWND hWnd = CreateWindow(L"HiddenD3DtoWPFWindow", L"HiddenD3DtoWPFWindow", WS_OVERLAPPEDWINDOW, 
				0, 0, 0, 0, NULL, NULL, wndClassEx.hInstance, NULL);

			m_pRenderer=new DirectX9Renderer(hWnd, d3dpp);
		}

		SceneGraphFactory::~SceneGraphFactory(void)
		{
			if(m_pRenderer)
				delete m_pRenderer;
		}

		MaterialNode^ SceneGraphFactory::CreateMaterial(void)
		{
			return gcnew MaterialNode(m_pRenderer);
		}

		TextureLayerNode^ SceneGraphFactory::CreateTextureLayer(void)
		{
			return gcnew TextureLayerNode(m_pRenderer);
		}

		TextureLayerNode^	SceneGraphFactory::CreateTextureLayer(TargetRenderAsset^ a_pTarget)
		{
			TextureLayerNode^	result = gcnew TextureLayerNode(m_pRenderer);
			DirectX9TextureLayer* unmanagedLayer=(DirectX9TextureLayer*)result->GetUnmanagedNode();
			unmanagedLayer->setTextureSource(0, a_pTarget->GetUnmanagedAsset()->getTexture());

			return result;
		}

		TextureLayerNode^	SceneGraphFactory::CreateTextureLayer(CudaRaytraceRenderAsset^ a_pTarget)
		{
			TextureAsset^ texture= gcnew TextureAsset(m_pRenderer);
			TextureAsset^ texture1= gcnew TextureAsset(m_pRenderer);
			TextureAsset^ texture2= gcnew TextureAsset(m_pRenderer);
			TextureAsset^ texture3= gcnew TextureAsset(m_pRenderer);

			TextureLayerNode^	result = gcnew TextureLayerNode(m_pRenderer);
			result->SetTexture(0, texture);

			a_pTarget->SetTextureTarget(0, texture);
			a_pTarget->SetTextureTarget(1, texture1);
			a_pTarget->SetTextureTarget(2, texture2);
			a_pTarget->SetTextureTarget(3, texture3);
			return result;
		}

		TextureLayerNode^	SceneGraphFactory::CreateTextureLayer(CudaVideoRenderAsset^ a_pTarget)
		{
			TextureLayerNode^	result = gcnew TextureLayerNode(m_pRenderer);
			DirectX9TextureLayer* unmanagedLayer=(DirectX9TextureLayer*)result->GetUnmanagedNode();

			CudaVideoRender* cudaVideoRender=a_pTarget->GetUnmanagedAsset();

			unmanagedLayer->setTextureSource(0, cudaVideoRender->getTextureTarget(0));
			return result;
		}

		TextureLayerNode^	SceneGraphFactory::CreateTextureLayer(RaytraceRenderAsset^ a_pTarget)
		{
			TextureLayerNode^	result = gcnew TextureLayerNode(m_pRenderer);
			DirectX9Texture* unmanagedTexture= new DirectX9Texture(m_pRenderer);
			//TODO: make target render to same texture always, maybe with a delegate
			unmanagedTexture->setGrid(a_pTarget->GetUnmanagedAsset()->getGrid());
			DirectX9TextureLayer* unmanagedLayer=(DirectX9TextureLayer*)result->GetUnmanagedNode();
			unmanagedLayer->setTextureSource(0, unmanagedTexture);
			return result;
		}

		TextureLayerNode^	SceneGraphFactory::CreateTextureLayer(TextureAsset^ a_pTexture, TextureAsset^ a_pTexture1)
		{
			TextureLayerNode^	result = gcnew TextureLayerNode(m_pRenderer);		
			result->SetTexture(0, a_pTexture);		
			return result;
		}

		DrawMockNode^ SceneGraphFactory::CreateDrawMock(void)
		{
			return gcnew DrawMockNode(m_pRenderer);
		}

		BaseNode^ SceneGraphFactory::CreateBaseNode(void)
		{
			return gcnew BaseNode();
		}

		ClearBufferNode^ SceneGraphFactory::CreateClearBuffer(void)
		{
			return gcnew ClearBufferNode(m_pRenderer);
		}

		TransformNode^	SceneGraphFactory::CreateTransform(void)
		{
			return gcnew TransformNode();
		}

		TrackballNode^	SceneGraphFactory::CreateTrackball(void)
		{
			return gcnew TrackballNode();
		}

		CameraAsset^	SceneGraphFactory::CreateCamera(void)
		{
			return gcnew CameraAsset();
		}

		EffectAsset^	SceneGraphFactory::CreateEffect(System::String^ a_sFilename)
		{
			return gcnew EffectAsset(m_pRenderer, a_sFilename);
		}

		SdkMeshAsset^	SceneGraphFactory::CreateSdkMesh(System::String^ a_sFilename)
		{
			return gcnew SdkMeshAsset(m_pRenderer, a_sFilename);
		}

		ObjMeshAsset^	SceneGraphFactory::CreateObjMesh(System::String^ a_sFilename)
		{
			return gcnew ObjMeshAsset(m_pRenderer, a_sFilename);
		}

		ViewRenderAsset^ SceneGraphFactory::CreateView(void)
		{
			return gcnew ViewRenderAsset(m_pRenderer);
		}

		TargetRenderAsset^	SceneGraphFactory::CreateRenderTargetAsset(void)
		{
			return gcnew TargetRenderAsset(m_pRenderer);
		}

		TargetRenderAsset^	SceneGraphFactory::CreateRenderTargetA8R8G8B8Asset(void)
		{
			return gcnew TargetRenderAsset(m_pRenderer, D3DFMT_A8R8G8B8);
		}

		RaytraceRenderAsset^	SceneGraphFactory::CreateRaytracer(void)
		{
			return gcnew RaytraceRenderAsset();
		}

		CudaRaytraceRenderAsset^	SceneGraphFactory::CreateCudaRaytraceRenderAsset(void)
		{
			return gcnew CudaRaytraceRenderAsset(m_pRenderer);
		}

		CudaVideoRenderAsset^	SceneGraphFactory::CreateCudaVideoRenderAsset(System::String^ a_sFilename)
		{
			CudaVideoRenderAsset^ cudaVideoRenderAsset=gcnew CudaVideoRenderAsset(m_pRenderer, a_sFilename);

			unsigned int width=cudaVideoRenderAsset->GetUnmanagedAsset()->getVideoWidth();
			unsigned int height=cudaVideoRenderAsset->GetUnmanagedAsset()->getVideoHeight();

			DirectX9Texture* directX9TextureField1;
			DirectX9Texture* directX9TextureField2;

			directX9TextureField1 = new DirectX9Texture(m_pRenderer, width, height, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8);
			directX9TextureField2 = new DirectX9Texture(m_pRenderer, width, height, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8);

			cudaVideoRenderAsset->GetUnmanagedAsset()->setTextureTarget(0, directX9TextureField1);
			cudaVideoRenderAsset->GetUnmanagedAsset()->setTextureTarget(1, directX9TextureField2);		

			//TODO: set textures as target (field1 and field2)
			return cudaVideoRenderAsset;
		}

		CudaFileRenderAsset^	SceneGraphFactory::CreateCudaFileRenderAsset(System::String^ a_sFilename)
		{
			CudaFileRenderAsset^ cudaFileRenderAsset=gcnew CudaFileRenderAsset(m_pRenderer, a_sFilename);

			//unsigned int width=1280;
			//unsigned int height=720;
			//directX9TextureField1 = new DirectX9Texture(m_pRenderer, width, height, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8);		
			//directX9TextureField1 = new DirectX9Texture(m_pRenderer, width, height, D3DUSAGE_DYNAMIC, D3DFMT_A32B32G32R32F);
			//directX9TextureField1 = new DirectX9Texture(m_pRenderer, width, height, D3DUSAGE_DYNAMIC, D3DFMT_YUY2);
			//D3DFMT_UYVY     D3DFMT_YUY2
			//directX9TextureField1->fillColorGardient();
			//cudaFileRenderAsset->GetUnmanagedAsset()->setTextureSource(0, directX9TextureField1);

			return cudaFileRenderAsset;
		}

		DirectShowFileRenderAsset^	SceneGraphFactory::CreateDirectShowFileRenderAsset(System::String^ a_sFilename){
			DirectShowFileRenderAsset^ directShowFileRenderAsset=gcnew DirectShowFileRenderAsset(a_sFilename);
			return directShowFileRenderAsset;
		}

		MediaFoundationFileRenderAsset^	SceneGraphFactory::CreateMediaFoundationFileRenderAsset(System::String^ a_sFilename, unsigned int width, unsigned int height){
			MediaFoundationFileRenderAsset^ mediaFoundationFileRenderAsset=gcnew MediaFoundationFileRenderAsset(a_sFilename, width, height);
			return mediaFoundationFileRenderAsset;
		}

		TextureAsset^		SceneGraphFactory::CreateTextureAsset(TargetRenderAsset^ a_pTarget)
		{
			TextureAsset^	result = gcnew TextureAsset(a_pTarget->GetUnmanagedAsset()->getTexture());
			return result;
		}

		TextureAsset^		SceneGraphFactory::CreateTextureAsset(TextureLayerNode^ a_pTextureLayerNode)
		{
			DirectX9TextureLayer* unmanagedLayer=(DirectX9TextureLayer*)a_pTextureLayerNode->GetUnmanagedNode();

			TextureAsset^	result = gcnew TextureAsset(unmanagedLayer->getTextureSource(0));
			return result;
		}

		TextureAsset^		SceneGraphFactory::CreateTextureAsset(CudaVideoRenderAsset^ a_pCudaVideoRenderAsset)
		{
			CudaVideoRender* unmanagedLayer=a_pCudaVideoRenderAsset->GetUnmanagedAsset();

			TextureAsset^	result = gcnew TextureAsset(unmanagedLayer->getTextureTarget(0));
			return result;
		}

		System::Drawing::Bitmap^		SceneGraphFactory::CreateBitmap(TextureAsset^ a_pTexture) {
			try {
				Grid<TheWhiteAmbit::Color>* pGrid = a_pTexture->GetUnmanagedAsset()->getGrid();

				// Retrieve the image.
				Bitmap^ image = gcnew Bitmap(pGrid->getWidth(), pGrid->getHeight(), Imaging::PixelFormat::Format32bppArgb );

				// Loop through the images pixels to reset color.
				for ( unsigned int x = 0; x < image->Width; x++ ) {
					for ( unsigned int y = 0; y < image->Height; y++ ) {
						//TODO: inspect all the twists in texture memory layout
						D3DXVECTOR4 vecColor=pGrid->getPixel( image->Width - x - 1, y);
						int color_r = max(0, min(255, (int)(vecColor.x*255.0f)));
						int color_g = max(0, min(255, (int)(vecColor.y*255.0f)));
						int color_b = max(0, min(255, (int)(vecColor.z*255.0f)));
						int color_a = max(0, min(255, (int)(vecColor.w*255.0f)));
						//color_a = 255 - color_a;
						//color_a = 255;
						System::Drawing::Color newColor = System::Drawing::Color::FromArgb( color_a, color_r, color_g, color_b );
						image->SetPixel( x, y, newColor );
					}
				}

				delete pGrid;
				return image;
			} catch (...) {
				return nullptr;
			}
		}

		GridAsset^	SceneGraphFactory::CreateGrid(TextureAsset^ a_pTexture) {
			try {
				Grid<TheWhiteAmbit::Color>* pGrid = a_pTexture->GetUnmanagedAsset()->getGrid();
				GridAsset^ image = gcnew GridAsset(pGrid);
				return image;
			} catch (...) {
				return nullptr;
			}
		}

		System::Drawing::Bitmap^		SceneGraphFactory::CreateBitmap(GridAsset^ a_pGrid) {
			try {
				Grid<TheWhiteAmbit::Color>* pGrid = a_pGrid->GetUnmanagedAsset();

				// Retrieve the image.
				Bitmap^ image = gcnew Bitmap(pGrid->getWidth(), pGrid->getHeight(), Imaging::PixelFormat::Format32bppArgb );

				// Loop through the images pixels to reset color.
				for ( unsigned int x = 0; x < image->Width; x++ ) {
					for ( unsigned int y = 0; y < image->Height; y++ ) {
						//TODO: inspect all the twists in texture memory layout
						TheWhiteAmbit::Color vecColor=pGrid->getPixel( image->Width - x - 1, y);
						int color_r = max(0, min(255, (int)(vecColor.x*255.0f)));
						int color_g = max(0, min(255, (int)(vecColor.y*255.0f)));
						int color_b = max(0, min(255, (int)(vecColor.z*255.0f)));
						int color_a = max(0, min(255, (int)(vecColor.w*255.0f)));
						//color_a = 255 - color_a;
						//color_a = 255;
						System::Drawing::Color newColor = System::Drawing::Color::FromArgb( color_a, color_r, color_g, color_b );
						image->SetPixel( x, y, newColor );
					}
				}					
				return image;
			} catch (...) {
				return nullptr;
			}
		}

		TextureAsset^	SceneGraphFactory::CreateTextureAsset(System::Drawing::Bitmap^ a_pBitmap) {
			Grid<TheWhiteAmbit::Color>* pGrid = new Grid<TheWhiteAmbit::Color>(a_pBitmap->Width, a_pBitmap->Height);

			// Loop through the images pixels to reset color.
			for ( unsigned int x = 0; x < a_pBitmap->Width; x++ ) {
				for ( unsigned int y = 0; y < a_pBitmap->Height; y++ ) {
					System::Drawing::Color newColor = a_pBitmap->GetPixel(x, y);
					TheWhiteAmbit::Color vecColor;
					vecColor.x = ((double)newColor.R/255.0);
					vecColor.y = (float)((double)newColor.G/255.0);
					vecColor.z = (float)((double)newColor.B/255.0);
					vecColor.w = (float)((double)newColor.A/255.0);
					pGrid->setPixel( x, y , vecColor);				
				}
			}
			DirectX9Texture* pTexture=new DirectX9Texture(m_pRenderer, a_pBitmap->Width, a_pBitmap->Height, D3DUSAGE_RENDERTARGET, D3DFMT_A32B32G32R32F);
			pTexture->setGrid(pGrid);
			delete pGrid;
			return gcnew TextureAsset(pTexture);
		}
	}
}