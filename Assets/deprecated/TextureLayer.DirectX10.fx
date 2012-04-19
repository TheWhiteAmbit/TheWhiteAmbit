//--------------------------------------------------------------------------------------
// File: MeshFromOBJ10.fx
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------




//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
float3 g_vMaterialAmbient   = float3( 0.2f, 0.2f, 0.2f );   // Material's ambient color
float3 g_vMaterialDiffuse   = float3( 0.8f, 0.8f, 0.8f );   // Material's diffuse color
float3 g_vMaterialSpecular  = float3( 1.0f, 1.0f, 1.0f );   // Material's specular color
float  g_fMaterialAlpha     = 1.0f;
int    g_nMaterialShininess = 32;

float3 g_vLightColor        = float3( 1.0f, 1.0f, 1.0f );     // Light color
float3 g_vLightPosition     = float3( 50.0f, 10.0f, 0.0f );   // Light position
float3 g_vCameraPosition    = float3( 5.0f, 5.0f, 5.0f );     // Light color

Texture2D g_MeshTexture;        // Color texture for mesh
Texture2D g_MeshTextureBump;        // Color texture for mesh

float  Time ;                // App's time in seconds
matrix   World ;               // World matrix
matrix   View ;               // World matrix
matrix   WorldViewProjection ; // World * View * Projection matrix

//--------------------------------------------------------------------------------------
// Texture sampler
//--------------------------------------------------------------------------------------
SamplerState samLinear
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

//--------------------------------------------------------------------------------------
// Vertex shader input structure
//--------------------------------------------------------------------------------------
struct VS_INPUT
{
    float4 vPosObject   : POSITION;
    float3 vNormalObject: NORMAL;
    float2 vTexUV       : TEXCOORD0;
};

//--------------------------------------------------------------------------------------
// Pixel shader input structure
//--------------------------------------------------------------------------------------
struct PS_INPUT
{
    float4 vPosProj : SV_POSITION;
    float4 vColor   : COLOR0;
    float2 vTexUV   : TEXCOORD0;
};


PS_INPUT VS( VS_INPUT input, uniform bool bSpecular )
{
    PS_INPUT output;
    output.vPosProj = float4(input.vPosObject.yxz,1);
	//output.vPosProj.z = output.vPosProj.x;
	//output.vPosProj = float4(0,0,0,1);
    output.vTexUV = input.vTexUV;
    output.vColor = float4(0, 1, 0, .5);
    return output;
}

float4 PS( PS_INPUT input, uniform bool bTexture ) : SV_Target
{
    float4 output = input.vColor;
    //output = g_MeshTexture.Sample( samLinear, input.vTexUV ).rgba;
    output = float4(1,1,0,.5);
    return output;
}

//--------------------------------------------------------------------------------------
// Techniques
//--------------------------------------------------------------------------------------
RasterizerState DisableCulling { CullMode = NONE; };
DepthStencilState DepthEnabling { DepthEnable = TRUE; };
DepthStencilState DepthDisabling { DepthEnable = FALSE;	DepthWriteMask = ZERO; };
BlendState DisableBlend { BlendEnable[0] = FALSE; };

technique10 NoSpecular
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_4_0, VS(false) ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, PS(false) ) );
		SetRasterizerState(DisableCulling);
		SetDepthStencilState(DepthDisabling, 0);
	    SetBlendState(DisableBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}
technique10 Specular
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_4_0, VS(true) ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, PS(false) ) );
		SetRasterizerState(DisableCulling);
		SetDepthStencilState(DepthDisabling, 0);
	    SetBlendState(DisableBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}
technique10 TexturedNoSpecular
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_4_0, VS(false) ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, PS(true) ) );
		SetRasterizerState(DisableCulling);
		SetDepthStencilState(DepthDisabling, 0);
	    SetBlendState(DisableBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}
technique10 TexturedSpecular
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_4_0, VS(true) ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, PS(true) ) );
		SetRasterizerState(DisableCulling);
		SetDepthStencilState(DepthDisabling, 0);
	    SetBlendState(DisableBlend, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}


