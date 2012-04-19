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
     vPositionOut = vPosProj ;
     vTexCoordOut = vTexCoordIn;
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
    bool isLed;
    if(vTexCoord.y < 0.52){    
        vTexCoord.x +=0.016f;
        vTexCoord.y +=0.016f;

        vTexCoord.x = -vTexCoord.x;
        vTexCoord.y = -vTexCoord.y;

        vTexCoord.x /=1.280f;
        vTexCoord.y /=1.024f;
        isLed=true;
    } else {
        vTexCoord.x = -vTexCoord.x;
        //vTexCoord.y = -vTexCoord.y;
        
        vTexCoord.x /=1.280f;
        vTexCoord.y /=1.024f;

        vTexCoord.y *= 16.0f/9.0f;
        vTexCoord.y -= .1f;
        isLed=false;
    }
        
        

    float4 color=tex2D( MeshTextureSampler, vTexCoord ).rgba;

    float truncX = (vTexCoord.x*-1280.0f-trunc(vTexCoord.x*-1280.0f)-.5f);
    float truncY = (vTexCoord.y*-1024.0f-trunc(vTexCoord.y*-1024.0f)-.5f);

    float ledHit= truncX * truncX + truncY * truncY;
    ledHit = sqrt(ledHit);
    
    float fade=min(1.0f, max(0.0f, (0.5f-ledHit)*3.0f))*2.0f;
    float dist=min(1.0f, max(0.0f, (vPositionIn.z/2.0f)-(7.0f/2.0f)));

    fade = min(1.0f, max(dist, fade));
    fade -= max(0.0f, dist*.35f);
        
    if(isLed){
        color.r*=fade;
        color.g*=fade;
        color.b*=fade;
        color.r = max(0.25f, color.r);
        color.g = max(0.25f, color.g);
        color.b = max(0.25f, color.b);
    } else {
        
    }

    color.a = 1.0f;

    //if(vTexCoord.y > -0.52){
    //    color.x = 1.0f;
    //    color.y = 0.0f;
    //    color.z = 0.0f;
    //} else {      
    //    color.x = 0.0f;
    //    color.y = 1.0f;
    //    color.z = 0.0f;
    //}
    
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
