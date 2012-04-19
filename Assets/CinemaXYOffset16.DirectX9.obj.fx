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
float    Z_NEAR : ZNear = 0.01f;          // App's z-Near value
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
float    GreyR : GreyR;
float    GreyG : GreyG;
float    GreyB : GreyB;
float    Size : Size;
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
};


//--------------------------------------------------------------------------------------
// Name: Projection
// Type: Vertex Shader Fragment
// Desc: Projection transform
//--------------------------------------------------------------------------------------
void Projection( float4 vPosObject: POSITION,
                 float4 vNormalObject: NORMAL,
                 float2 vTexCoordIn: TEXCOORD0,
                 out float4 vPosProj: POSITION,
                 out float2 vTexCoordOut: TEXCOORD0,
                 out float4 vPositionOut: TEXCOORD1,
                 uniform bool bSpecular
                )
{
	vPosProj = mul( vPosObject, WorldViewProjection );
	
    	vTexCoordIn.x +=0.015f;
    	vTexCoordIn.y +=0.015f;

     //to unmap uv
     float invTime=(1.0f-Time)*2.0f;
     //const float aspectInv=9.0f/16.0f;
     const float aspectInv=3.0f/4.0f;
     //const float aspectInv=4.0f/5.0f;
     vPositionOut = vPosProj;
     vPosProj = float4(-vPosProj.x*Time + invTime*(vTexCoordIn.x-0.5f)*aspectInv*2.0f, vPosProj.y*Time + invTime*(vTexCoordIn.y-0.5f)*2.0f, vPosProj.z * Time, vPosProj.w*Time + invTime);

     vTexCoordOut = vTexCoordIn;
     vTexCoordOut.y = -vTexCoordOut.y;	
     //vTexCoordOut.y /=0.768f;	
     //vTexCoordOut.x /=1.024f;	

     //vTexCoordOut.y /=0.576;	
     //vTexCoordOut.x /=0.720;	

     //vTexCoordOut.y /=0.720;
     //vTexCoordOut.x /=1.280;

     vTexCoordOut.y /=1.024f;	
     vTexCoordOut.x /=1.280;
}


//--------------------------------------------------------------------------------------
// Name: Lighting
// Type: Pixel Shader
// Desc: Compute lighting and modulate the texture
//--------------------------------------------------------------------------------------
void Lighting( float2 vTexCoord: TEXCOORD0,
               float4 vPositionIn: TEXCOORD1,
               out float4 vColorOut: COLOR0,
               uniform bool bTexture )
{  
    float4 color=tex2D( MeshTextureSampler, vTexCoord ).rgba;

    //float truncX = (vTexCoord.x*1280.0f-trunc(vTexCoord.x*1280.0f)-.5f);
    //float truncY = (vTexCoord.y*-720.0f-trunc(vTexCoord.y*-720.0f)-.5f);
    //float truncX = (vTexCoord.x*1024.0f-trunc(vTexCoord.x*1024.0f)-.5f);
    //float truncY = (vTexCoord.y*-768.0f-trunc(vTexCoord.y*-768.0f)-.5f);
    float truncX = (vTexCoord.x*1280.0f-trunc(vTexCoord.x*1280.0f)-.5f);
    float truncY = (vTexCoord.y*-1024.0f-trunc(vTexCoord.y*-1024.0f)-.5f);

    float ledHit= truncX * truncX + truncY * truncY;
    ledHit = sqrt(ledHit);
    
    float fade=min(1.0f, max(0.0f, (Size-ledHit)*3.0f))*2.0f;
    fade = min(1.0f, fade);

    color.r *= fade;
    color.g *= fade;
    color.b *= fade;
    
    color.r = (GreyR + color.r)/ (1.0 + GreyR);
    color.g = (GreyG + color.g)/ (1.0 + GreyG);
    color.b = (GreyB + color.b)/ (1.0 + GreyB);

    color.a = 1.0f;    
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
        //CullMode = CW;
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
       //CullMode = CW;
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
       //CullMode = CW;
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
       //CullMode = CW;
	  //AlphaBlendEnable = true;
	  AlphaTestEnable = false;
	  SrcBlend = SrcAlpha;
	  DestBlend = InvSrcAlpha;
        VertexShader = compile vs_2_0 Projection(false);    
        PixelShader = compile ps_2_0 Lighting(true);    
    }
}
