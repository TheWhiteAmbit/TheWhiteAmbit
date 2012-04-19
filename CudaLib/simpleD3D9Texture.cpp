/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

 /* This example demonstrates how to use the CUDA Direct3D bindings to 
  * transfer data between CUDA and DX9 2D, CubeMap, and Volume Textures.
  */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <d3dx9.h>

// includes, project

#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>

#include <cuda_d3d9_interop.h>
#include <cutil_inline.h>

#include <cassert>

bool                  g_bDone = false;
IDirect3D9		    * g_pD3D; // Used to create the D3DDevice
IDirect3DDevice9    * g_pD3DDevice;

D3DDISPLAYMODE        g_d3ddm;    
D3DPRESENT_PARAMETERS g_d3dpp;    

bool                  g_bWindowed    = true;
bool                  g_bDeviceLost  = false;

const unsigned int    g_WindowWidth  = 720;
const unsigned int    g_WindowHeight = 720;

// Data structure for 2D texture shared between DX9 and CUDA
struct
{
	IDirect3DTexture9* pTexture;
	int width;
	int height;	
} g_texture_2d;

// Data structure for cube texture shared between DX9 and CUDA
struct
{
	IDirect3DCubeTexture9* pTexture;
	int size;
} g_texture_cube;

// Data structure for volume textures shared between DX9 and CUDA
struct
{
	IDirect3DVolumeTexture9* pTexture;
	int width;
	int height;
	int depth;
} g_texture_vol;

// The CUDA kernel launchers that get called
extern "C" 
{
	bool cuda_texture_2d(void* surface, size_t width, size_t height, size_t pitch, float t);
	bool cuda_texture_cube(void* surface, int width, int height, size_t pitch, int face, float t);
	bool cuda_texture_volume(void* surface, int width, int height, int depth, size_t pitch, size_t pitchslice);
}


//-----------------------------------------------------------------------------
// Forward declarations
//-----------------------------------------------------------------------------
HRESULT InitD3D9( HWND hWnd );
HRESULT InitCUDA( );
HRESULT ReleaseCUDA( );
HRESULT InitTextures( );
HRESULT ReleaseTextures();
HRESULT RegisterD3D9ResourceWithCUDA();
HRESULT DeviceLostHandler();

void RunKernels();
HRESULT DrawScene();
void Cleanup();
void RunCUDA();

LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	//
	// create window
	//
    // Register the window class
#if 1
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, MsgProc, 0L, 0L,
                      GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
                      L"CUDA SDK", NULL };
    RegisterClassEx( &wc );

    // Create the application's window
    HWND hWnd = CreateWindow( wc.lpszClassName, L"CUDA D3D9 Texture InterOP",
                              WS_OVERLAPPEDWINDOW, 0, 0, g_WindowWidth, g_WindowHeight,
                              NULL, NULL, wc.hInstance, NULL );
#else
	static WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, MsgProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, "CudaD3D9Tex", NULL };
    RegisterClassEx(&wc);
    HWND hWnd = CreateWindow(
		"CudaD3D9Tex", "CUDA D3D9 Texture Interop", 
		WS_OVERLAPPEDWINDOW, 
		0, 0, 800, 320, 
		GetDesktopWindow(), 
		NULL, 
		wc.hInstance, 
		NULL );
#endif

    ShowWindow(hWnd, SW_SHOWDEFAULT);
    UpdateWindow(hWnd);

    // Initialize Direct3D
    if( SUCCEEDED( InitD3D9(hWnd) ) &&
        SUCCEEDED( InitCUDA() ) &&
		SUCCEEDED( InitTextures() ) )
	{
        if (!g_bDeviceLost) {
            RegisterD3D9ResourceWithCUDA();
        }
	}

	//
	// the main loop
	//
    while(false == g_bDone) 
	{
        RunCUDA();
        DrawScene();

		//
		// handle I/O
		//
        MSG msg;
        ZeroMemory( &msg, sizeof(msg) );
        while( msg.message!=WM_QUIT )
        {
            if( PeekMessage( &msg, NULL, 0U, 0U, PM_REMOVE ) )
            {
                TranslateMessage( &msg );
                DispatchMessage( &msg );
            }
            else
			{
                RunCUDA();
                DrawScene();
			}
        }

    };

	// Unregister windows class
	UnregisterClass( wc.lpszClassName, wc.hInstance );

	//
	// and exit
	//
	return 0;
}

//-----------------------------------------------------------------------------
// Name: InitD3D9()
// Desc: Initializes Direct3D9
//-----------------------------------------------------------------------------
HRESULT InitD3D9(HWND hWnd) 
{
    // Create the D3D object.
    if( NULL == ( g_pD3D = Direct3DCreate9( D3D_SDK_VERSION ) ) )
        return E_FAIL;

    // Get primary display identifier
    D3DADAPTER_IDENTIFIER9 adapterId;
    
    // Find the first CUDA capable device
    unsigned int adapter;
    for(adapter = 0; adapter < g_pD3D->GetAdapterCount(); adapter++)
    {
        D3DADAPTER_IDENTIFIER9 ident;
        int device;
        g_pD3D->GetAdapterIdentifier(adapter, 0, &adapterId);
        cudaD3D9GetDevice(&device, ident.DeviceName);
        if (cudaSuccess == cudaGetLastError() ) break;
    }
    // we check to make sure we have found a cuda-compatible device to work on
    if(adapter != g_pD3D->GetAdapterCount() )
    {
        // print some error message here
		printf("(adapter=%d, GetAdapterCount()=%d)\n", adapter, g_pD3D->GetAdapterCount());
    }

	// Create the D3D Display Device
    RECT                  rc;       GetClientRect(hWnd,&rc);
    D3DDISPLAYMODE        d3ddm;    g_pD3D->GetAdapterDisplayMode(D3DADAPTER_DEFAULT, &d3ddm);
    D3DPRESENT_PARAMETERS d3dpp;    ZeroMemory( &d3dpp, sizeof(d3dpp) );
    d3dpp.Windowed               = TRUE;
    d3dpp.BackBufferCount        = 1;
    d3dpp.SwapEffect             = D3DSWAPEFFECT_DISCARD;
    d3dpp.hDeviceWindow          = hWnd;
    d3dpp.BackBufferWidth	     = rc.right  - rc.left;
    d3dpp.BackBufferHeight       = rc.bottom - rc.top;
    d3dpp.BackBufferFormat       = d3ddm.Format;
    
	if (FAILED (g_pD3D->CreateDevice (D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hWnd, 
									  D3DCREATE_HARDWARE_VERTEXPROCESSING, 
									  &d3dpp, &g_pD3DDevice) ))
		return E_FAIL;	

	// We clear the back buffer
	g_pD3DDevice->BeginScene();
	g_pD3DDevice->Clear(0, NULL, D3DCLEAR_TARGET, 0, 1.0f, 0);	
	g_pD3DDevice->EndScene();

	return S_OK;
}

HRESULT InitCUDA()
{
    printf("InitCUDA() g_pD3DDevice = %p\n", g_pD3DDevice);

    // Now we need to bind a CUDA context to the DX9 device
	// This is the CUDA 2.0 DX9 interface (required for Windows XP and Vista)
	cudaD3D9SetDirect3DDevice(g_pD3DDevice);
    cutilCheckMsg("cudaD3D9SetDirect3DDevice failed");

    return S_OK;
}

HRESULT ReleaseCUDA()
{
    // Uninitialize CUDA
    cudaThreadExit();
    cutilCheckMsg("cudaThreadExit failed");
    return S_OK;
}

HRESULT RegisterD3D9ResourceWithCUDA()
{
	// register the Direct3D resources that we'll use
	// we'll read to and write from g_texture_2d, so don't set any special map flags for it
    cudaD3D9RegisterResource(g_texture_2d.pTexture, cudaD3D9RegisterFlagsNone);
	cutilCheckMsg("cudaD3D9RegisterResource (g_texture_2d) failed");
	// Initialize this texture to be black
	{
        cutilSafeCallNoSync ( cudaD3D9MapResources (1, (IDirect3DResource9 **)&g_texture_2d.pTexture) );
		void* data;
		size_t size;
		cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&data, g_texture_2d.pTexture, 0, 0) );
		cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedSize(&size, g_texture_2d.pTexture, 0, 0) );
		cudaMemset(data, 0, size);
        cutilSafeCallNoSync ( cudaD3D9UnmapResources (1, (IDirect3DResource9 **)&g_texture_2d.pTexture) );
	}

	// we'll be overwriting g_texture_cube in full every frame, so map it as WriteDiscard
    cudaD3D9RegisterResource(g_texture_cube.pTexture, cudaD3D9RegisterFlagsNone);
	cutilCheckMsg("cudaD3D9RegisterResource (g_texture_cube) failed");
    cudaD3D9ResourceSetMapFlags(g_texture_cube.pTexture, cudaD3D9MapFlagsWriteDiscard);
	cutilCheckMsg("cudaD3D9ResourceSetMapFlags (g_texture_cube) failed");
	// Initialize this texture to be black
    for (unsigned int face = 0; face < 6; ++face)
	{
		cutilSafeCallNoSync ( cudaD3D9MapResources (1, (IDirect3DResource9 **)&g_texture_cube.pTexture) );
		void* data;
		size_t size;
        cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&data, g_texture_cube.pTexture, face, 0) );
        cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedSize(&size, g_texture_cube.pTexture, face, 0) );
		cudaMemset(data, 0, size);
		cutilSafeCallNoSync ( cudaD3D9UnmapResources (1, (IDirect3DResource9 **)&g_texture_cube.pTexture) );
	}

	// we'll be overwriting g_texture_vol in full every frame, so map it as WriteDiscard
    cudaD3D9RegisterResource(g_texture_vol.pTexture, cudaD3D9RegisterFlagsNone);
	cutilCheckMsg("cudaD3D9RegisterResource (g_texture_vol) failed");
	cudaD3D9ResourceSetMapFlags(g_texture_vol.pTexture, cudaD3D9MapFlagsWriteDiscard);
	cutilCheckMsg("cudaD3D9ResourceSetMapFlags (g_texture_vol) failed");

    return S_OK;
}

//-----------------------------------------------------------------------------
// Name: InitTextures()
// Desc: Initializes Direct3D Textures (allocation and initialization)
//-----------------------------------------------------------------------------
HRESULT InitTextures()
{
	//
	// create the D3D resources we'll be using
	//

	// 2D texture
	g_texture_2d.width  = 256;
	g_texture_2d.height = 256;
	if (FAILED(g_pD3DDevice->CreateTexture(g_texture_2d.width, g_texture_2d.height, 1, 0,
                                           D3DFMT_A32B32G32R32F, D3DPOOL_DEFAULT, &g_texture_2d.pTexture, NULL) ))
	{
		return E_FAIL;
	}

	// cube texture
	g_texture_cube.size = 64;
	if (FAILED(g_pD3DDevice->CreateCubeTexture(g_texture_cube.size, 1, 0, 
												D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, 
												&g_texture_cube.pTexture, NULL) ))
	{
		return E_FAIL;
	}

	// 3D texture
	g_texture_vol.width  = 64;
	g_texture_vol.height = 64;
	g_texture_vol.depth  = 32;
	
	if (FAILED(g_pD3DDevice->CreateVolumeTexture(	g_texture_vol.width, g_texture_vol.height, 
													g_texture_vol.depth, 1, 0, D3DFMT_A8R8G8B8, 
													D3DPOOL_DEFAULT, &g_texture_vol.pTexture, NULL) ))
	{
		return E_FAIL;
	}

	return S_OK;
}

//-----------------------------------------------------------------------------
// Name: ReleaseTextures()
// Desc: Release Direct3D Textures (free-ing)
//-----------------------------------------------------------------------------
HRESULT ReleaseTextures()
{
	// unregister the Cuda resources
	cudaD3D9UnregisterResource(g_texture_2d.pTexture);
	cutilCheckMsg("cudaD3D9UnregisterResource (g_texture_2d) failed");

	cudaD3D9UnregisterResource(g_texture_cube.pTexture);
	cutilCheckMsg("cudaD3D9UnregisterResource (g_texture_cube) failed");

	cudaD3D9UnregisterResource(g_texture_vol.pTexture);
	cutilCheckMsg("cudaD3D9UnregisterResource (g_texture_vol) failed");

	//
	// clean up Direct3D
	// 
	{
		// release the resources we created
		g_texture_2d.pTexture->Release();
		g_texture_cube.pTexture->Release();
		g_texture_vol.pTexture->Release();
    }

    return S_OK;
}


////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void RunKernels()
{
	static float t = 0.0f;

	// populate the 2d texture
	{
		void* pData;
		size_t pitch;
		cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&pData, g_texture_2d.pTexture, 0, 0) );
		cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch, NULL, g_texture_2d.pTexture, 0, 0) );
        cuda_texture_2d(pData, g_texture_2d.width, g_texture_2d.height, pitch, t);
	}

	// populate the faces of the cube map
	for (int face = 0; face < 6; ++face)
	{
		void* pData;
		size_t pitch;
		cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&pData, g_texture_cube.pTexture, face, 0) );
		cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch, NULL, g_texture_cube.pTexture, face, 0) );
        cuda_texture_cube(pData, g_texture_cube.size, g_texture_cube.size, pitch, face, t);
	}

	// populate the volume texture
	{
		void* pData;
		size_t pitch;
		size_t pitchSlice;
		cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&pData, g_texture_vol.pTexture, 0, 0) );
		cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch, &pitchSlice, g_texture_vol.pTexture, 0, 0) );
        cuda_texture_volume(pData, g_texture_vol.width, g_texture_vol.height, g_texture_vol.depth, pitch, pitchSlice);
	}

	t += 0.1f;
}

////////////////////////////////////////////////////////////////////////////////
//! RestoreContextResources
//    - this function restores all of the CUDA/D3D resources and contexts
////////////////////////////////////////////////////////////////////////////////
HRESULT RestoreContextResources()
{
    // Reinitialize D3D9 resources, CUDA resources/contexts
    InitCUDA();
    InitTextures();
    RegisterD3D9ResourceWithCUDA();

    return S_OK;
}


////////////////////////////////////////////////////////////////////////////////
//! DeviceLostHandler
//    - this function handles reseting and initialization of the D3D device
//      in the event this Device gets Lost
////////////////////////////////////////////////////////////////////////////////
HRESULT DeviceLostHandler()
{
    HRESULT hr = S_OK;

    fprintf(stderr, "-> Starting DeviceLostHandler() \n");
    // test the cooperative level to see if it's okay
    // to render
    if (FAILED(hr = g_pD3DDevice->TestCooperativeLevel()))
    {
        fprintf(stderr, "TestCooperativeLevel = %08x failed, will attempt to reset\n", hr);

        // if the device was truly lost, (i.e., a fullscreen device just lost focus), wait
        // until we g_et it back

        if (hr == D3DERR_DEVICELOST) {
            fprintf(stderr, "TestCooperativeLevel = %08x DeviceLost, will retry next call\n", hr);
            return S_OK;
        }

        // eventually, we will g_et this return value,
        // indicating that we can now reset the device
        if (hr == D3DERR_DEVICENOTRESET)
        {
            fprintf(stderr, "TestCooperativeLevel = %08x will try to RESET the device\n", hr);
            // if we are windowed, read the desktop mode and use the same format for 
            // the back buffer; this effectively turns off color conversion

            if (g_bWindowed)
            {
                g_pD3D->GetAdapterDisplayMode( D3DADAPTER_DEFAULT, &g_d3ddm );
                g_d3dpp.BackBufferFormat = g_d3ddm.Format;
            }

            // now try to reset the device
            if (FAILED(hr = g_pD3DDevice->Reset(&g_d3dpp))) {
                fprintf(stderr, "TestCooperativeLevel = %08x RESET device FAILED!\n", hr);
                return hr;
            } else {
                fprintf(stderr, "TestCooperativeLevel = %08x RESET device SUCCESS!\n", hr);

                // This is a common function we use to restore all hardware resources/state
                RestoreContextResources();

                fprintf(stderr, "TestCooperativeLevel = %08x INIT device SUCCESS!\n", hr);

                // we have acquired the device
                g_bDeviceLost = false;
            }
        }
    }
    return hr;
}

////////////////////////////////////////////////////////////////////////////////
//! Draw the final result on the screen
////////////////////////////////////////////////////////////////////////////////
HRESULT DrawScene()
{
    HRESULT hr = S_OK;

    if (g_bDeviceLost) 
    {
        if ( FAILED(hr = DeviceLostHandler()) ) {
            fprintf(stderr, "DeviceLostHandler FAILED! returned %08x\n", hr);
            return hr;
        }
    }

    if (!g_bDeviceLost) 
    {
	    //
	    // we will use this index and vertex data throughout
	    //
	    unsigned int IB[6] = 
	    {
		    0,1,2,
		    0,2,3,
	    };
	    struct VertexStruct
	    {
		    float position[3];
		    float texture[3];
	    };

	    // 
	    // initialize the scene
	    //
	    D3DVIEWPORT9 viewport_window = {0, 0, 672, 192, -1, 1};
	    g_pD3DDevice->SetViewport(&viewport_window);
	    g_pD3DDevice->BeginScene();
	    g_pD3DDevice->Clear(0, NULL, D3DCLEAR_TARGET, 0, 1.0f, 0);	
	    g_pD3DDevice->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
	    g_pD3DDevice->SetRenderState(D3DRS_LIGHTING, FALSE);
	    g_pD3DDevice->SetFVF(D3DFVF_XYZ|D3DFVF_TEX1|D3DFVF_TEXCOORDSIZE3(0));

	    //
	    // draw the 2d texture
	    //
	    VertexStruct VB[4] = 
	    {
		    {  {-1,-1,0,}, {0,0,0,},  },
		    {  { 1,-1,0,}, {1,0,0,},  },
		    {  { 1, 1,0,}, {1,1,0,},  },
		    {  {-1, 1,0,}, {0,1,0,},  },
	    };
	    D3DVIEWPORT9 viewport = {32, 32, 256, 256, -1, 1};
	    g_pD3DDevice->SetViewport(&viewport);
	    g_pD3DDevice->SetTexture(0,g_texture_2d.pTexture);
        g_pD3DDevice->DrawIndexedPrimitiveUP(D3DPT_TRIANGLELIST, 0, 4, 2, IB, D3DFMT_INDEX32, VB, sizeof(VertexStruct) );

	    //
	    // draw the Z-positive side of the cube texture
	    //
	    VertexStruct VB_Zpos[4] = 
	    {
		    {  {-1,-1,0,}, {-1,-1, 0.5f,},  },
		    {  { 1,-1,0,}, { 1,-1, 0.5f,},  },
		    {  { 1, 1,0,}, { 1, 1, 0.5f,},  },
		    {  {-1, 1,0,}, {-1, 1, 0.5f,},  },
	    };
	    viewport.Y += viewport.Height + 32;
	    g_pD3DDevice->SetViewport(&viewport);
	    g_pD3DDevice->SetTexture(0,g_texture_cube.pTexture);
        g_pD3DDevice->DrawIndexedPrimitiveUP(D3DPT_TRIANGLELIST, 0, 4, 2, IB, D3DFMT_INDEX32, VB_Zpos, sizeof(VertexStruct) );

	    //
	    // draw the Z-negative side of the cube texture
	    //
	    VertexStruct VB_Zneg[4] = 
	    {
		    {  {-1,-1,0,}, { 1,-1,-0.5f,},  },
		    {  { 1,-1,0,}, {-1,-1,-0.5f,},  },
		    {  { 1, 1,0,}, {-1, 1,-0.5f,},  },
		    {  {-1, 1,0,}, { 1, 1,-0.5f,},  },
	    };
	    viewport.X += viewport.Width + 32;
	    g_pD3DDevice->SetViewport(&viewport);
	    g_pD3DDevice->SetTexture(0,g_texture_cube.pTexture);
        g_pD3DDevice->DrawIndexedPrimitiveUP(D3DPT_TRIANGLELIST, 0, 4, 2, IB, D3DFMT_INDEX32, VB_Zneg, sizeof(VertexStruct) );

	    //
	    // draw a slice the volume texture
	    //
	    VertexStruct VB_Zslice[4] = 
	    {
		    {  {-1,-1,0,}, {0,0,0,},  },
		    {  { 1,-1,0,}, {1,0,0,},  },
		    {  { 1, 1,0,}, {1,1,1,},  },
		    {  {-1, 1,0,}, {0,1,1,},  },
	    };	
	    viewport.Y -= viewport.Height + 32;
	    g_pD3DDevice->SetViewport(&viewport);
	    g_pD3DDevice->SetTexture(0,g_texture_vol.pTexture);
        g_pD3DDevice->DrawIndexedPrimitiveUP(D3DPT_TRIANGLELIST, 0, 4, 2, IB, D3DFMT_INDEX32, VB_Zslice, sizeof(VertexStruct) );

	    //
	    // end the scene
	    //
	    g_pD3DDevice->EndScene();
	    hr = g_pD3DDevice->Present(NULL, NULL, NULL, NULL);

        if (hr == D3DERR_DEVICELOST) {
            fprintf(stderr, "DrawScene Present = %08x detected D3D DeviceLost\n", hr);
            g_bDeviceLost = true;

            ReleaseTextures();
            ReleaseCUDA();
        }
    }
    return hr;
}

//-----------------------------------------------------------------------------
// Name: Cleanup()
// Desc: Releases all previously initialized objects
//-----------------------------------------------------------------------------
void Cleanup()
{
    ReleaseTextures();

    ReleaseCUDA();

	{
		// destroy the D3D device
		g_pD3DDevice->Release();
	}
}

//-----------------------------------------------------------------------------
// Name: RunCUDA()
// Desc: Launches the CUDA kernels to fill in the texture data
//-----------------------------------------------------------------------------
void RunCUDA()
{
	//
	// map the resources we've registered so we can access them in Cuda
	// - it is most efficient to map and unmap all resources in a single call,
	//   and to have the map/unmap calls be the boundary between using the GPU
	//   for Direct3D and Cuda
	//

    if (!g_bDeviceLost) {
	    IDirect3DResource9* ppResources[3] = 
	    {
		    g_texture_2d.pTexture,
		    g_texture_cube.pTexture,
		    g_texture_vol.pTexture,
	    };
	    cudaD3D9MapResources(3, ppResources);
	    cutilCheckMsg("cudaD3D9MapResources(3) failed");

	    //
	    // run kernels which will populate the contents of those textures
	    //
	    RunKernels();

	    //
	    // unmap the resources
	    //
	    cudaD3D9UnmapResources(3, ppResources);
	    cutilCheckMsg("cudaD3D9UnmapResources(3) failed");
    }
}

//-----------------------------------------------------------------------------
// Name: MsgProc()
// Desc: The window's message handler
//-----------------------------------------------------------------------------
static LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch(msg)
    {
        case WM_KEYDOWN:
            if(wParam==VK_ESCAPE) 
			{
				g_bDone = true;
                Cleanup();
	            PostQuitMessage(0);
				return 0;
			}
            break;
        case WM_DESTROY:
			g_bDone = true;
            Cleanup();
            PostQuitMessage(0);
            return 0;
        case WM_PAINT:
            ValidateRect(hWnd, NULL);
            return 0;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

