//--------------------------------------------------------------------------------------
// File: MeshFromOBJ.fx
//
// The effect file for the MeshFromOBJ sample.  
// 
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

float3 g_vMaterialAmbient : Ambient = float3( 0.2f, 0.2f, 0.2f );   // Material's ambient color
float3 g_vMaterialDiffuse : Diffuse = float3( 0.8f, 0.8f, 0.8f );   // Material's diffuse color
float3 g_vMaterialSpecular : Specular = float3( 1.0f, 1.0f, 1.0f );  // Material's specular color
float  g_fMaterialAlpha : Opacity = 1.0f;
int    g_nMaterialShininess : SpecularPower = 32;

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------           
float    Z_NEAR : ZNear = 0.01f;          // App's z-Near value
float    Time : Time = 0.0f;          // App's time in seconds
float4x4 World : World;          // World matrix
float4x4 View : View;          // World matrix
float4x4 WorldViewProjection : WorldViewProjection; // World * View * Projection matrix

Texture2D g_MeshTexture : Texture;        // Color texture for mesh

//--------------------------------------------------------------------------------------
// Texture samplers
//--------------------------------------------------------------------------------------
sampler MeshTextureSampler = 
sampler_state
{
    Texture = <g_MeshTexture>;    
    MipFilter = LINEAR;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
};


void Projection( float4 vPosObject: POSITION,
				float3 vNormalObject: NORMAL,
				float4 vTexCoordIn: TEXCOORD0,
				out float4 vPosProjOut: POSITION,
				out float4 vTexCoordOut: TEXCOORD0,
				out float4 vColorOut: TEXCOORD1,
				out float4 vPosOut: TEXCOORD2
				)
{
	vPosOut = vPosObject;
	float4 vPosProj = mul( vPosObject, WorldViewProjection );

	//to unmap uv
	float invTime=(1.0f-Time)*2.0f;
	vPosProj = float4(vPosProj.x*Time + invTime*(vTexCoordIn.x-.5), vPosProj.y*Time + invTime*(vTexCoordIn.y-.5), vPosProj.z * Time, vPosProj.w*Time + invTime*.5);
	vPosProjOut = vPosProj;
	vColorOut.rgb = vNormalObject;
	vTexCoordOut = vTexCoordIn;
}

void Lighting(float4 vTexCoordIn: TEXCOORD0,
			  float4 vNormIn: TEXCOORD1,
			  float4 vPosIn: TEXCOORD2,
			  out float4 vColorOut: COLOR0
			  )
{  
	float3 normal=vNormIn.rgb;
	normal=mul(normal, (float3x3)World);
	normal=mul(normal, (float3x3)View);    
	vColorOut.rgb = normal.rgb*.5+float3(.5,.5,.5);
	//vColorOut.rgb = vPosIn.xyz*.5+float3(.5,.5,.5);
	//vColorOut.rgb = vTexCoordIn.xyz*.5+float3(.5,.5,.5);
	

	float4 vPosProj = vPosIn;
	vPosProj = mul(vPosProj, World);
	vPosProj = mul(vPosProj, View);
	vColorOut.a = (Z_NEAR/vPosProj.z);
	//vColorOut.a = 0.0f;
	//vColorOut.rgb = float3(1.0f/vPosProj.x,1.0f/vPosProj.y,1.0f/vPosProj.z);
	//vColorOut.rgb = vPosProj.xyz*.5+float3(.5,.5,.5);
}


//--------------------------------------------------------------------------------------
// Techniques
//--------------------------------------------------------------------------------------
technique Specular
{
	pass P0
	{
		//CullMode = CW;
		CullMode = None;
		AlphaTestEnable = false;
		//DepthMask = true;
		VertexShader = compile vs_2_0 Projection();    
		PixelShader = compile ps_2_0 Lighting();    
	}
}

technique NoSpecular
{
	pass P0
	{
		//CullMode = None;
		VertexShader = compile vs_2_0 Projection();    
		PixelShader = compile ps_2_0 Lighting();    
	}
}

technique TexturedSpecular
{
	pass P0
	{
		//CullMode = None;
		VertexShader = compile vs_2_0 Projection();    
		PixelShader = compile ps_2_0 Lighting();    
	}
}

technique TexturedNoSpecular
{
	pass P0
	{
		//CullMode = None;
		VertexShader = compile vs_2_0 Projection();    
		PixelShader = compile ps_2_0 Lighting();    
	}
}