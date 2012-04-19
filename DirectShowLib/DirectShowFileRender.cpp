

#include <iostream>
#include <time.h>
#include <comdef.h>
#include <atlbase.h>
#include <wchar.h>

//#include <streams.h>
//#include <amstream.h>
//#include <dvdmedia.h>
//#include <mmsystem.h>
//#include <atlbase.h>
//#include <stdio.h>
//#include <mtype.h>
//#include <wxdebug.h>
//#include <reftime.h>

#include <wmcodecdsp.h>


#include "DirectShowFileRender.h"


// {CCCE52FD-02CB-482c-AC81-1E55EF1D61EE}
static const GUID CLSID_H264DecFilter = 
{ 0xccce52fd, 0x2cb, 0x482c, { 0xac, 0x81, 0x1e, 0x55, 0xef, 0x1d, 0x61, 0xee } };

// {936E6340-19A8-4a58-92AE-695FD64B9418}
static const GUID CLSID_MPEG2DecFilter = 
{ 0x936e6340, 0x19a8, 0x4a58, { 0x92, 0xae, 0x69, 0x5f, 0xd6, 0x4b, 0x94, 0x18 } };

// {F58D5C1C-8EC7-4e74-B3A9-CED73B25F4A1}
static const GUID CLSID_VC1DecFilter = 
{ 0xf58d5c1c, 0x8ec7, 0x4e74, { 0xb3, 0xa9, 0xce, 0xd7, 0x3b, 0x25, 0xf4, 0xa1 } };

// {71183C45-F4FA-4b10-9E04-F9040CB19139}
static const GUID CLSID_H264EncFilter = 
{ 0x71183c45, 0xf4fa, 0x4b10, { 0x9e, 0x4, 0xf9, 0x4, 0xc, 0xb1, 0x91, 0x39 } };

// {F0EAA393-2ACD-4cbe-8F4D-990DEB6C67E6}
static const GUID CLSID_MPEG2EncFilter = 
{ 0xf0eaa393, 0x2acd, 0x4cbe, { 0x8f, 0x4d, 0x99, 0xd, 0xeb, 0x6c, 0x67, 0xe6 } };

// {281D4741-787E-4a2d-B518-69C4CB1D7227}
static const GUID CLSID_VC1EncFilter = 
{ 0x281d4741, 0x787e, 0x4a2d, { 0xb5, 0x18, 0x69, 0xc4, 0xcb, 0x1d, 0x72, 0x27 } };

// {2eeb4adf-4578-4d10-bca7-bb955f56320a}
static const CLSID CLSID_CWMADecMediaObject = 
{ 0x2eeb4adf, 0x4578, 0x4d10, { 0xbc, 0xa7, 0xbb, 0x95, 0x5f, 0x56, 0x32, 0x0a } };

// {41E5E4D6-7635-4c43-8A06-DD856470856F}
static const GUID CLSID_MPEG2SplitterFilter =
{ 0x41e5e4d6, 0x7635, 0x4c43, { 0x8a, 0x6, 0xdd, 0x85, 0x64, 0x70, 0x85, 0x6f } };

// {A2A6B846-D118-4300-AE07-F31860887BC2}
static const GUID CLSID_MP4SplitterFilter = 
{ 0xa2a6b846, 0xd118, 0x4300, { 0xae, 0x7, 0xf3, 0x18, 0x60, 0x88, 0x7b, 0xc2 } };

//AUDIO FILTERS

// {E7FACCFD-9148-4871-B302-60D7A1FC6270}
static const GUID CLSID_AC3DecFilter = 
{ 0xe7faccfd, 0x9148, 0x4871, { 0xb3, 0x2, 0x60, 0xd7, 0xa1, 0xfc, 0x62, 0x70 } };

// {06079E43-C107-4b50-8450-3C09FF5E832E}
static const GUID CLSID_MP3DecFilter = 
{ 0x6079e43, 0xc107, 0x4b50, { 0x84, 0x50, 0x3c, 0x9, 0xff, 0x5e, 0x83, 0x2e } };

// {8DA364BE-DF1D-43f9-9A86-CC06F53C082C}
static const GUID CLSID_AACDecFilter = 
{ 0x8da364be, 0xdf1d, 0x43f9, { 0x9a, 0x86, 0xcc, 0x6, 0xf5, 0x3c, 0x8, 0x2c } };

//aac encoder filter GIUDS
const GUID CLSID_AACEncFilter =
{ 0xe51ef49d, 0xddb0, 0x4874, { 0xa8, 0x73, 0xc5, 0x10, 0x1, 0x71, 0x14, 0x6f } };

// {CECE2B60-4954-41ac-8971-ECD874A4C368}
const GUID CLSID_MP3EncFilter =
{ 0xcece2b60, 0x4954, 0x41ac, { 0x89, 0x71, 0xec, 0xd8, 0x74, 0xa4, 0xc3, 0x68 } };

// MUXER FILTERS

// {CB488050-23B8-411d-B861-D00BA44B8D02}
static const GUID CLSID_MP4MuxerFilter = 
{ 0xcb488050, 0x23b8, 0x411d, { 0xb8, 0x61, 0xd0, 0xb, 0xa4, 0x4b, 0x8d, 0x2 } };

// {AF76B26C-ECDE-4515-BB41-C149BBC362CE}
static const GUID CLSID_MPEG2MuxerFilter = 
{ 0xaf76b26c, 0xecde, 0x4515, { 0xbb, 0x41, 0xc1, 0x49, 0xbb, 0xc3, 0x62, 0xce } };



namespace TheWhiteAmbit {

	std::string FormatSize( DWORD dwFileSize );
	std::string InsertSeparator( DWORD dwNumber );


	void ShowPropertyPage(	IBaseFilter *pFilter )
	{
		/* Obtain the filter's IBaseFilter interface. (Not shown) */
		ISpecifyPropertyPages *pProp;
		HRESULT hr = pFilter->QueryInterface(IID_ISpecifyPropertyPages, (void **)&pProp);
		if (SUCCEEDED(hr)) 
		{
			// Get the filter's name and IUnknown pointer.
			FILTER_INFO FilterInfo;
			hr = pFilter->QueryFilterInfo(&FilterInfo); 
			IUnknown *pFilterUnk;
			pFilter->QueryInterface(IID_IUnknown, (void **)&pFilterUnk);

			// Show the page. 
			CAUUID caGUID;
			pProp->GetPages(&caGUID);
			pProp->Release();
			OleCreatePropertyFrame(
				NULL,                   // Parent window handle
				0, 0,                   // Reserved
				FilterInfo.achName,     // Caption for the dialog box
				1,                      // Number of objects (just the filter)
				&pFilterUnk,            // Array of object pointers. 
				caGUID.cElems,          // Number of property pages
				caGUID.pElems,          // Array of property page CLSIDs
				0,                      // Locale identifier
				0, NULL                 // Reserved
				);

			// Clean up.
			pFilterUnk->Release();
			FilterInfo.pGraph->Release(); 
			CoTaskMemFree(caGUID.pElems);
		}
	}




	/// Create a filter by category and name. Will enumerate all filters
	/// of the given category and return the filter whose name matches,
	/// if any.
	///
	/// @param Name of filter to create.
	/// @param Filter Will receive the pointer to the interface
	/// for the created filter.
	/// @param FilterCategory Filter category.
	///
	/// @return S_OK if successful.
	HRESULT CreateFilter(const WCHAR *Name, REFCLSID FilterCategory, GUID* guid)
	{
		HRESULT hr=E_FAIL;    

		// Create the system device enumerator.
		CComPtr<ICreateDevEnum> devenum;
		hr = devenum.CoCreateInstance(CLSID_SystemDeviceEnum);
		if (FAILED(hr))
			return hr;    

		// Create an enumerator for this category.
		CComPtr<IEnumMoniker> classenum;
		hr = devenum->CreateClassEnumerator(FilterCategory, &classenum, 0);
		if (FAILED(hr))
			return hr;    

		// Find the filter that matches the name given.
		CComVariant name(Name);
		CComPtr<IMoniker> moniker;
		while (classenum->Next(1, &moniker, 0) == S_OK)
		{
			CComPtr<IPropertyBag> properties;
			hr = moniker->BindToStorage(0, 0, IID_IPropertyBag, (void **)&properties);
			if (FAILED(hr))
				return hr;    

			CComVariant friendlyname;
			hr = properties->Read(L"FriendlyName", &friendlyname, 0);
			if (FAILED(hr))
				return hr;    

			if (name == friendlyname)
			{
				IBaseFilter *Filter;
				hr = moniker->BindToObject(0, 0, IID_IBaseFilter, (void **)&Filter);
				if (FAILED(hr))
					return hr;

				hr = Filter->GetClassID(guid);
				if (FAILED(hr))
					return hr;
			}

			moniker.Release();
		}    

		// Couldn't find a matching filter.
		return hr;
	}


	HRESULT AddFilterByCLSID(
		IGraphBuilder *pGraph,  // Pointer to the Filter Graph Manager.
		const GUID& clsid,      // CLSID of the filter to create.
		LPCWSTR wszName,        // A name for the filter.
		IBaseFilter **ppF)      // Receives a pointer to the filter.
	{
		if (!pGraph || ! ppF) return E_POINTER;
		*ppF = 0;
		IBaseFilter *pF = 0;
		HRESULT hr = CoCreateInstance(clsid, 0, CLSCTX_INPROC_SERVER,
			IID_IBaseFilter, reinterpret_cast<void**>(&pF));
		if (SUCCEEDED(hr))
		{
			hr = pGraph->AddFilter(pF, wszName);
			if (SUCCEEDED(hr))
				*ppF = pF;
			else
				pF->Release();
		}
		return hr;
	}


	HRESULT GetUnconnectedPin(
		IBaseFilter *pFilter,   // Pointer to the filter.
		PIN_DIRECTION PinDir,   // Direction of the pin to find.
		IPin **ppPin)           // Receives a pointer to the pin.
	{
		*ppPin = 0;
		IEnumPins *pEnum = 0;
		IPin *pPin = 0;
		HRESULT hr = pFilter->EnumPins(&pEnum);
		if (FAILED(hr))
		{
			return hr;
		}
		while (pEnum->Next(1, &pPin, NULL) == S_OK)
		{
			PIN_DIRECTION ThisPinDir;
			pPin->QueryDirection(&ThisPinDir);
			if (ThisPinDir == PinDir)
			{
				IPin *pTmp = 0;
				hr = pPin->ConnectedTo(&pTmp);
				if (SUCCEEDED(hr))  // Already connected, not the pin we want.
				{
					pTmp->Release();
				}
				else  // Unconnected, this is the pin we want.
				{
					pEnum->Release();
					*ppPin = pPin;
					return S_OK;
				}
			}
			pPin->Release();
		}
		pEnum->Release();
		// Did not find a matching pin.
		return E_FAIL;
	}



	HRESULT ConnectFilters(
		IGraphBuilder *pGraph, // Filter Graph Manager.
		IPin *pOut,            // Output pin on the upstream filter.
		IBaseFilter *pDest)    // Downstream filter.
	{
		if ((pGraph == NULL) || (pOut == NULL) || (pDest == NULL))
		{
			return E_POINTER;
		}
		//#ifdef DEBUG
		//        PIN_DIRECTION PinDir;
		//        pOut->QueryDirection(&PinDir);
		//        _ASSERTE(PinDir == PINDIR_OUTPUT);
		//#endif

		// Find an input pin on the downstream filter.
		IPin *pIn = 0;
		HRESULT hr = GetUnconnectedPin(pDest, PINDIR_INPUT, &pIn);
		if (FAILED(hr))
		{
			return hr;
		}
		// Try to connect them.
		hr = pGraph->Connect(pOut, pIn);
		pIn->Release();
		return hr;
	}

	HRESULT ConnectFilters(
		IGraphBuilder *pGraph, 
		IBaseFilter *pSrc, 
		IBaseFilter *pDest)
	{
		if ((pGraph == NULL) || (pSrc == NULL) || (pDest == NULL))
		{
			return E_POINTER;
		}

		// Find an output pin on the first filter.
		IPin *pOut = 0;
		HRESULT hr = GetUnconnectedPin(pSrc, PINDIR_OUTPUT, &pOut);
		if (FAILED(hr)) 
		{
			return hr;
		}
		hr = ConnectFilters(pGraph, pOut, pDest);
		pOut->Release();
		return hr;
	}

	HRESULT ConnectFiltersDirect(
		IGraphBuilder *pGraph, // Filter Graph Manager.
		IPin *pOut,            // Output pin on the upstream filter.
		IBaseFilter *pDest,
		AM_MEDIA_TYPE *pmt)    // Downstream filter.
	{
		if ((pGraph == NULL) || (pOut == NULL) || (pDest == NULL))
		{
			return E_POINTER;
		}
		//#ifdef DEBUG
		//        PIN_DIRECTION PinDir;
		//        pOut->QueryDirection(&PinDir);
		//        _ASSERTE(PinDir == PINDIR_OUTPUT);
		//#endif

		// Find an input pin on the downstream filter.
		IPin *pIn = 0;
		HRESULT hr = GetUnconnectedPin(pDest, PINDIR_INPUT, &pIn);
		if (FAILED(hr))
		{
			return hr;
		}
		// Try to connect them.
		hr = pGraph->ConnectDirect(pOut, pIn, pmt);
		pIn->Release();
		return hr;
	}

	HRESULT ConnectFiltersDirect(
		IGraphBuilder *pGraph, 
		IBaseFilter *pSrc, 
		IBaseFilter *pDest,
		AM_MEDIA_TYPE *pmt)
	{
		if ((pGraph == NULL) || (pSrc == NULL) || (pDest == NULL))
		{
			return E_POINTER;
		}

		// Find an output pin on the first filter.
		IPin *pOut = 0;
		HRESULT hr = GetUnconnectedPin(pSrc, PINDIR_OUTPUT, &pOut);
		if (FAILED(hr)) 
		{
			return hr;
		}
		hr = ConnectFiltersDirect(pGraph, pOut, pDest, pmt);
		pOut->Release();
		return hr;
	}


	HRESULT ConnectFiltersAllPins(
		IGraphBuilder *pGraph, // Filter Graph Manager.
		IPin *pOut,            // Output pin on the upstream filter.
		IBaseFilter *pDest)    // Downstream filter.
	{
		if ((pGraph == NULL) || (pOut == NULL) || (pDest == NULL))
		{
			return E_POINTER;
		}

		// Find an input pin on the downstream filter.
		IEnumPins *pEnum = 0;
		IPin *pIn = 0;

		HRESULT hr = pDest->EnumPins(&pEnum);
		if (FAILED(hr))
		{
			return hr;
		}
		while (pEnum->Next(1, &pIn, NULL) == S_OK)
		{
			PIN_DIRECTION ThisPinDir;
			pIn->QueryDirection(&ThisPinDir);
			if (ThisPinDir == PINDIR_INPUT)
			{
				IPin *pTmp = 0;
				hr = pIn->ConnectedTo(&pTmp);
				if (SUCCEEDED(hr))  // Already connected, not the pin we want.
				{
					pTmp->Release();
				}
				else  // Unconnected, this is the pin we want.
				{

					// Try to connect them.
					hr = pGraph->Connect(pOut, pIn);								
					if (SUCCEEDED(hr))
					{
						pIn->Release();
						pEnum->Release();
						return S_OK;
					}
				}
			}
			pIn->Release();
		}
		pEnum->Release();
		// Did not find a matching pin.
		return E_FAIL;
	}


	HRESULT ConnectFiltersAllPins(
		IGraphBuilder *pGraph, 
		IBaseFilter *pSrc, 
		IBaseFilter *pDest)
	{
		if ((pGraph == NULL) || (pSrc == NULL) || (pDest == NULL))
		{
			return E_POINTER;
		}

		// Find an output pin on the first filter.

		IEnumPins *pEnum = 0;
		IPin *pOut = 0;
		HRESULT hr = pSrc->EnumPins(&pEnum);
		if (FAILED(hr))
		{
			return hr;
		}
		while (pEnum->Next(1, &pOut, NULL) == S_OK)
		{
			PIN_DIRECTION ThisPinDir;
			pOut->QueryDirection(&ThisPinDir);
			if (ThisPinDir == PINDIR_OUTPUT)
			{
				IPin *pTmp = 0;
				hr = pOut->ConnectedTo(&pTmp);
				if (SUCCEEDED(hr))  // Already connected, not the pin we want.
				{
					pTmp->Release();
				}
				else  // Unconnected, this is the pin we want.
				{
					hr = ConnectFiltersAllPins(pGraph, pOut, pDest);				
					if (SUCCEEDED(hr))
					{
						pOut->Release();
						pEnum->Release();
						return hr;
					}
				}
			}
			pOut->Release();
		}
		pEnum->Release();
		// Did not find a matching pin.
		return E_FAIL;
	}


#ifdef _DEBUG
#define REGISTER_FILTERGRAPH    
#endif
#ifdef REGISTER_FILTERGRAPH

	//-----------------------------------------------------------------------------
	// Running Object Table functions: Used to debug. By registering the graph
	// in the running object table, GraphEdit is able to connect to the running
	// graph. This code should be removed before the application is shipped in
	// order to avoid third parties from spying on your graph.
	//-----------------------------------------------------------------------------
	HRESULT AddToRot(IUnknown *pUnkGraph, DWORD *pdwRegister) 
	{
		IMoniker * pMoniker = NULL;
		IRunningObjectTable *pROT = NULL;

		if (FAILED(GetRunningObjectTable(0, &pROT))) 
		{
			return E_FAIL;
		}

		const size_t STRING_LENGTH = 256;

		WCHAR wsz[STRING_LENGTH];

		//StringCchPrintfW(
		//     wsz, STRING_LENGTH, 
		//     L"FilterGraph %08x pid %08x", 
		//     (DWORD_PTR)pUnkGraph, 
		//     GetCurrentProcessId()
		//     );

		HRESULT hr = CreateItemMoniker(L"!", L"TheWhiteAmbit FilterGraph", &pMoniker);
		if (SUCCEEDED(hr)) 
		{
			hr = pROT->Register(ROTFLAGS_REGISTRATIONKEEPSALIVE, pUnkGraph,
				pMoniker, pdwRegister);
			pMoniker->Release();
		}
		pROT->Release();

		return hr;
	}

	void RemoveFromRot(DWORD pdwRegister)
	{
		IRunningObjectTable *pROT;
		if (SUCCEEDED(GetRunningObjectTable(0, &pROT))) {
			pROT->Revoke(pdwRegister);
			pROT->Release();
		}
	}
#endif

	DirectShowFileRender::DirectShowFileRender(LPCWSTR a_pFilename)
	{	
		CoInitialize( 0 );
		{
			hStdout = GetStdHandle( STD_OUTPUT_HANDLE );
			pFilterGraph = NULL;
			pGraphBuilder = NULL;
			pMediaControl = NULL;
			pSeek = NULL;
			pMediaEvent = NULL;
			pMediaFilter = NULL;
			pEncoder = NULL;
			pFileWriter = NULL;
			pSink = NULL;

#ifdef USE_MUXER
			pMuxingFilter = NULL;
#endif		
#ifdef USE_FILESOURCE
			pFileSource = NULL;
#endif
#ifndef USE_FILESOURCE
			pFrameSource = NULL;
#endif


			HRESULT hr = CoCreateInstance( CLSID_FilterGraph, NULL, CLSCTX_INPROC_SERVER, IID_IGraphBuilder, (void **)&pFilterGraph );
			if( FAILED( hr ) )
				wprintf( L"Could not create Filter Graph." );		

			hr = pFilterGraph->QueryInterface( IID_IGraphBuilder, (void **)&pGraphBuilder );
			if( FAILED( hr ) )
				wprintf( L"Could not get IGraphBuilder interface." );

#ifdef REGISTER_FILTERGRAPH
			// Register the graph in the Running Object Table (for debug purposes)
			hr = AddToRot(pGraphBuilder, &dwRegister);
			if( FAILED( hr ))
				wprintf( L"Could not register graph." );
#endif


			GUID clsid_codec;
			//std::wstring codec_wstr(codec_str);
			//hr = CreateFilter(codec_wstr.c_str(), CLSID_LegacyAmFilterCategory, &clsid_codec);
			//hr = CreateFilter(L"ffdshow video encoder", CLSID_VideoCompressorCategory, &clsid_codec);
			//hr = CreateFilter(L"Microsoft MPEG-2 Video Encoder", CLSID_LegacyAmFilterCategory, &clsid_codec);
			//hr = CreateFilter(L"Microsoft MPEG-2 Video Encoder", CLSID_MediaEncoderCategory, &clsid_codec);
			//hr = CreateFilter(L"DV Video Encoder", CLSID_VideoCompressorCategory, &clsid_codec);
			//hr = CreateFilter(L"eWorker QuickTime Encoder", CLSID_VideoCompressorCategory, &clsid_codec);
			//hr = CreateFilter(L"MJPEG Compressor", CLSID_VideoCompressorCategory, &clsid_codec);
			//CLSID_VideoCompressorCategory
			//CLSID_MediaEncoderCategory
			//CLSID_MediaMultiplexerCategory
			//CLSID_LegacyAmFilterCategory
			//CLSID_CMPEG2EncoderVideoDS
			//CLSID_CMpeg4EncMediaObject


			//-sf wildlife.wmv -of wildlife.avi -dr 2000 -cd "ffdshow video encoder" -mx "AVI Mux"
			//-sf wildlife.wmv -of wildlife.avi -dr 2000 -cd "Microsoft MPEG-2 Video Encoder" -mx "AVI Mux"

			if( FAILED( hr ) )
				wprintf( L"Could not find encoder filter." );

			//hr = AddFilterByCLSID(pGraphBuilder, clsid_codec, L"encoder", &pEncoder);
			hr = AddFilterByCLSID(pGraphBuilder, CLSID_MPEG2EncFilter, L"encoder", &pEncoder);
			if( FAILED( hr ) )
				wprintf( L"Could not add encoder filter." );

			//hr = CreateFilter(L"File writer", CLSID_LegacyAmFilterCategory, &pFileWriter);
			hr = AddFilterByCLSID(pGraphBuilder, CLSID_FileWriter, L"File writer", &pFileWriter);
			if( FAILED( hr ) )
				wprintf( L"Could not add File writer filter." );

			hr = pFileWriter->QueryInterface( IID_IFileSinkFilter, (void **)&pSink );
			if( FAILED( hr ) )
				wprintf( L"Could not get IFileSinkFilter interface." );

			hr = pSink->SetFileName( a_pFilename, NULL );
			if( FAILED( hr ) )
				wprintf( L"Could not set output filename." );

			//pFrameSource = new CFrameSource(NULL, &hr);								
			pFrameSource = (CFrameSource*) CFrameSource::CreateInstance(NULL, &hr);
			if (FAILED(hr)) 
				wprintf( L"Could not create frame source filter." );

			hr = pGraphBuilder->AddFilter( pFrameSource, L"source"  );
			if( FAILED( hr ) )
				wprintf( L"Could not add frame source filter." );

#ifndef USE_FILESOURCE
			hr = AddFilterByCLSID(pGraphBuilder, CLSID_Colour, L"Colorconverter", &pColorConverter);
			if( FAILED( hr ) )
				wprintf( L"Could not add Colorconversion filter." );
			//CLSID_AVIDec
			//CLSID_VideoRenderer
			//CLSID_Colour

			hr = ConnectFilters(pGraphBuilder, pFrameSource, pColorConverter);
			if( FAILED( hr ) )
				wprintf( L"Could not connect Framesource to Colorconverter filter." );

			hr = ConnectFilters(pGraphBuilder, pColorConverter, pEncoder);
			if( FAILED( hr ) )
				wprintf( L"Could not connect Colorconverter to encoder filter." );
			/////////////////////
			////CMediaType MediaType;
			////MediaType.InitMediaType();
			////MediaType.majortype = MEDIATYPE_Video;
			////MediaType.subtype = MEDIASUBTYPE_RGB24;
			////hr = ConnectFiltersDirect(pGraphBuilder, pFrameSource, pEncoder, &MediaType);
			//hr = ConnectFilters(pGraphBuilder, pFrameSource, pEncoder);
			//if( FAILED( hr ) )
			//	wprintf( L"Could not connect Framesource to encoder filter." );
			/////////////////////
			//hr = AddFilterByCLSID(pGraphBuilder, CLSID_VideoRenderer, L"VideoRenderer", &pColorConverter);
			//if( FAILED( hr ) )
			//	wprintf( L"Could not add VideoRenderer filter." );
			//hr = ConnectFilters(pGraphBuilder, pFrameSource, pColorConverter);
			//if( FAILED( hr ) )
			//	wprintf( L"Could not connect Framesource to VideoRenderer filter." );

#endif
#ifdef USE_FILESOURCE
			hr = pGraphBuilder->AddSourceFilter( a_pFilename, L"source", &pFileSource );
			if( FAILED( hr ) )
				wprintf( L"Could not add source file filter." );	

			hr = ConnectFilters(pGraphBuilder, pFileSource, pEncoder);
			if( FAILED( hr ) )
				wprintf( L"Could not connect File source to encoder filter." );

			//hr = pGraphBuilder->RenderFile( source_file.c_str(), NULL );
			//if( FAILED( hr ) )
			//	wprintf( L"Could not render source file." );
#endif

#ifndef USE_MUXER
			hr = ConnectFilters(pGraphBuilder, pEncoder, pFileWriter);
			if( FAILED( hr ) )
				wprintf( L"Could not connect Encoder to File writer filter." );		
#endif
#ifdef USE_MUXER
			GUID clsid_mux;
			std::wstring mux_wstr(mux_str);
			hr = CreateFilter(mux_wstr.c_str(), CLSID_MediaMultiplexerCategory, &clsid_mux);
			if( FAILED( hr ) )
				wprintf( L"Could not find mux(ing) filter." );

			hr = AddFilterByCLSID(pGraphBuilder, clsid_mux, L"muxing", &pMuxingFilter);
			if( FAILED( hr ) )
				wprintf( L"Could not add mux(ing) filter." );

			//hr = AddFilterByCLSID(pGraphBuilder, CLSID_AviDest, L"AVI Mux", &pMuxingFilter);
			//hr = AddFilterByCLSID(pGraphBuilder, CLSID_DVMux, L"DV Muxer", &pMuxingFilter);

			if( FAILED( hr ) )
				wprintf( L"Could not add AVI Mux filter." );

			hr = ConnectFilters(pGraphBuilder, pEncoder, pMuxingFilter);
			if( FAILED( hr ) )
				wprintf( L"Could not connect Encoder to Mux filter." );

			hr = ConnectFilters(pGraphBuilder, pMuxingFilter, pFileWriter);
			if( FAILED( hr ) )
				wprintf( L"Could not connect Mux to File writer filter." );
#endif

			hr = pFilterGraph->QueryInterface( IID_IMediaControl, (void **)&pMediaControl );
			if( FAILED( hr ))
				wprintf( L"Could not get IMediaControl interface." );

			total_source_frames = 0;
			hr = pFilterGraph->QueryInterface( IID_IMediaSeeking, (void **)&pSeek );
			if( SUCCEEDED( hr ))
			{
				hr = pSeek->SetTimeFormat( &TIME_FORMAT_FRAME );
				if( SUCCEEDED( hr ) )
					hr = pSeek->GetDuration( &total_source_frames );
			}

			hr = pFilterGraph->QueryInterface( IID_IMediaEvent, (void**)&pMediaEvent );
			if( FAILED( hr ))
				wprintf( L"Could not get IMediaEvent interface." );

			hr = pFilterGraph->QueryInterface( IID_IMediaFilter, (void**)&pMediaFilter );
			if( FAILED( hr ) )
				wprintf( L"Could not query MediaFilter." );	

			hr = pMediaFilter->SetSyncSource( NULL );

			ShowPropertyPage(pEncoder);

			hr = pMediaControl->Run();
			if( FAILED( hr ))
				wprintf( L"Could not run graph." );
		}
	}

	DirectShowFileRender::~DirectShowFileRender(void)
	{
		{		
#ifdef REGISTER_FILTERGRAPH
			// Pull graph from Running Object Table (Debug)
			RemoveFromRot(dwRegister);
#endif
			if( pMediaControl )
			{
				pMediaControl->Stop();
				pMediaControl->Release();
			}

			if( pEncoder ) 
				pEncoder->Release();
			if( pMediaEvent )	
				pMediaEvent->Release();
			if( pMediaFilter )	
				pMediaFilter->Release();
			if( pFilterGraph )	
				pFilterGraph->Release();
			if( pGraphBuilder )	
				pGraphBuilder->Release();
			if( pSink )
				pSink->Release();
			if( pSeek )
				pSeek->Release();

			//if( pFrameSource )
			//	pFrameSource->Release();
		}
		CoUninitialize();
	}

	void DirectShowFileRender::present(int effect) {
		//int passnum=-1;
		//int inframenumber=0;
		//int outframenumber=0;

		//__int64 totaltime, passtime, output_file_size = 0, temp_file_size = 0;
		////PassType passtype;

		//CONSOLE_SCREEN_BUFFER_INFO csbiInfo;
		//csbiInfo.dwCursorPosition.X = 0;
		//csbiInfo.dwCursorPosition.Y = 0;

		//long nEventCode = 0;
		//int curr_pcnt_complete = 0;
		//int prev_pcnt_complete = 0;

		//std::wcout << "elapsed time              : " << std::endl;			
		//std::wcout << "source frame              : " << std::endl;
		//std::wcout << "output frame              : " << std::endl;
		//std::wcout << "output file size          : " << std::endl;

		//time_t current_pass_start_time;
		//time_t start_time;

		//while( nEventCode != EC_COMPLETE )
		//{
		//	int y = 0, x = 28;
		//	passnum=-1;

		//	if( passnum > -1 )
		//	{
		//		time_t now;
		//		time( &now );
		//		time_t totaltime = now - start_time;
		//		TCHAR elapsed_time[ 9 ];
		//		struct tm* total = localtime( &totaltime );					
		//		csbiInfo.dwCursorPosition.X = x;
		//		csbiInfo.dwCursorPosition.Y = y;
		//		SetConsoleCursorPosition( hStdout, csbiInfo.dwCursorPosition );
		//		std::wcout << elapsed_time << std::endl;
		//	}

		//	csbiInfo.dwCursorPosition.X = x;
		//	csbiInfo.dwCursorPosition.Y = ++y;
		//	SetConsoleCursorPosition( hStdout, csbiInfo.dwCursorPosition );
		//	std::wcout << inframenumber;
		//	if( total_source_frames )
		//		std::wcout << " ( of " << total_source_frames << " )                     ";
		//	std::wcout << std::endl;

		//	csbiInfo.dwCursorPosition.X = x;
		//	csbiInfo.dwCursorPosition.Y = ++y;
		//	SetConsoleCursorPosition( hStdout, csbiInfo.dwCursorPosition );
		//	std::wcout << outframenumber << "          " << std::endl;

		//	csbiInfo.dwCursorPosition.X = x;
		//	csbiInfo.dwCursorPosition.Y = ++y;
		//	SetConsoleCursorPosition( hStdout, csbiInfo.dwCursorPosition );

		//	hr = pMediaEvent->WaitForCompletion( 200, &nEventCode );
		//}
	}

	void DirectShowFileRender::setFrameBuffer(Grid<Color>* a_pGrid){
		pFrameSource->setFrameBuffer(a_pGrid);
	}
}