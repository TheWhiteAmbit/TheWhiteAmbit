//--------------------------------------------------------------------------------------
// File: MeshFromOBJ.fx
//
// The effect file for the MeshFromOBJ sample.  
// 
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
float3 g_vMaterialAmbient : Ambient = float3( 0.2f, 0.2f, 0.2f );   // Material's ambient color
float3 g_vMaterialDiffuse : Diffuse = float3( 0.8f, 0.8f, 0.8f );   // Material's diffuse color
float3 g_vMaterialSpecular : Specular = float3( 1.0f, 1.0f, 1.0f );  // Material's specular color
float  g_fMaterialAlpha : Opacity = 1.0f;
int    g_nMaterialShininess : SpecularPower = 32;

float3 g_vLightColor : LightColor = float3( 1.0f, 1.0f, 1.0f );        // Light color
float3 g_vLightPosition : LightPosition = float3( .71f, 0.0f, -.71f );   // Light position
float3 g_vCameraPosition : CameraPosition = float3( 5.0f, 5.0f, 5.0f );   // Camera position

texture   g_MeshTexture : Texture;   // Color texture for mesh
texture   g_MeshTextureBump : TextureBump;   // Color texture for mesh
            
float    Time : Time;            // App's time in seconds
float4x4 World : World;          // World matrix
float4x4 View : View;          // World matrix
float4x4 WorldViewProjection : WorldViewProjection; // World * View * Projection matrix



//--------------------------------------------------------------------------------------
// Texture samplers
//--------------------------------------------------------------------------------------
sampler MeshTextureSampler = 
sampler_state
{
    Texture = <g_MeshTexture>;    
    MipFilter = NONE;
    MinFilter = NONE;
    MagFilter = NONE;
    AddressU = CLAMP; 
    AddressV = CLAMP;
};


//--------------------------------------------------------------------------------------
// Name: Projection
// Type: Vertex Shader Fragment
// Desc: Projection transform
//--------------------------------------------------------------------------------------
void Projection( float4 vPosObject: POSITION,
                 float3 vNormalObject: NORMAL,
                 float2 vTexCoordIn: TEXCOORD0,
                 out float4 vPosProj: POSITION,
                 out float2 vTexCoordOut: TEXCOORD0,
                 out float4 vColorOut: COLOR0,
                 uniform bool bSpecular
                )
{
   // Transform the position into world space for lighting, and projected space
    vPosProj = vPosObject;
    vTexCoordOut = vTexCoordIn;
}



//--------------------------------------------------------------------------------------
// Name: Lighting
// Type: Pixel Shader
// Desc: Compute lighting and modulate the texture
//--------------------------------------------------------------------------------------
void Lighting( float2 vTexCoord: TEXCOORD0,
               float4 vColorIn: COLOR0,
               out float4 vColorOut: COLOR0,
               uniform bool bTexture )
{  
    // Sample and modulate the texture
	float4 color=tex2D( MeshTextureSampler, vTexCoord ).rgba;
	//color.rgb=color.rgb*(1.0f-color.a)*1.0f+(1.0f-color.a);
	//color.rgb=color.rgb*(1.0f-color.a)*1.0f;
	//color.r=(1.0f-color.a)*2.0f;
	//color.g=(1.0f-color.a)*2.0f;
	//color.b += .125f;
	//color.g *= 1.1f;
	//color.r *= 1.1f;
	//color.a=0.0f;
	//color.r=(1.0f-color.r);
	//color.g=(1.0f-color.g);
	//color.b=(1.0f-color.b);
	//color.a=1.0f;
    vColorOut = color;
}


//--------------------------------------------------------------------------------------
// Techniques
//--------------------------------------------------------------------------------------
technique Specular
{
    pass P0
    {
        CullMode = None;
	  //AlphaBlendEnable = true;
	  AlphaTestEnable = false;
	  SrcBlend = SrcAlpha;
	  DestBlend = InvSrcAlpha;
        VertexShader = compile vs_2_0 Projection(true);    
        PixelShader = compile ps_2_0 Lighting(false);    
    }
}

technique NoSpecular
{
    pass P0
    {
        CullMode = None;
	  //AlphaBlendEnable = true;
	  AlphaTestEnable = false;
	  SrcBlend = SrcAlpha;
	  DestBlend = InvSrcAlpha;
        VertexShader = compile vs_2_0 Projection(false);    
        PixelShader = compile ps_2_0 Lighting(false);    
    }
}

technique TexturedSpecular
{
    pass P0
    {
        CullMode = None;
	  //AlphaBlendEnable = true;
	  AlphaTestEnable = false;
	  SrcBlend = SrcAlpha;
	  DestBlend = InvSrcAlpha;
        VertexShader = compile vs_2_0 Projection(true);    
        PixelShader = compile ps_2_0 Lighting(true);    
    }
}

technique TexturedNoSpecular
{
    pass P0
    {
        CullMode = None;
	  //AlphaBlendEnable = true;
	  AlphaTestEnable = false;
	  SrcBlend = SrcAlpha;
	  DestBlend = InvSrcAlpha;
        VertexShader = compile vs_2_0 Projection(false);    
        PixelShader = compile ps_2_0 Lighting(true);    
    }
}