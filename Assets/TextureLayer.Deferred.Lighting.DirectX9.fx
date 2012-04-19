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

//float3 g_vMaterialAmbient : Ambient = float3( 0.5f, 0.0f, 0.0f );   // Material's ambient color
//float3 g_vMaterialDiffuse : Diffuse = float3( -0.5f, 0.0f, 1.0f );   // Material's diffuse color
//float3 g_vMaterialSpecular : Specular = float3( 1.0f, 1.0f, 0.5f );  // Material's specular color

//float3 g_vMaterialAmbient : Ambient = float3( .9f, 1.0f, .8f );   // Material's ambient color
//float3 g_vMaterialDiffuse : Diffuse = float3( .9f, .7f, 1.5 );   // Material's diffuse color
//float3 g_vMaterialSpecular : Specular = float3( 1.0f, 1.0f, -.2f );  // Material's specular color

//float3 g_vMaterialAmbient : Ambient = float3( .9f, 1.0f, .8f );   // Material's ambient color
//float3 g_vMaterialDiffuse : Diffuse = float3( .9f, .7f, 1.5 );   // Material's diffuse color
//float3 g_vMaterialSpecular : Specular = float3( 1.0f, 1.0f, -.2f );  // Material's specular color

//float3 g_vMaterialAmbient : Ambient = float3( 0.3f, 0.4f, 0.3f );   // Material's ambient color
//float3 g_vMaterialDiffuse : Diffuse = float3( 0.0f, 0.0f, 1.0f );   // Material's diffuse color
//float3 g_vMaterialSpecular : Specular = float3( 1.0f, 0.5f, 0.0f );  // Material's specular color

//float3 g_vMaterialAmbient : Ambient = float3( 1.0f, 1.0f, 1.0f );   // Material's ambient color
//float3 g_vMaterialDiffuse : Diffuse = float3( -0.5f, -0.5f, 0.5f );   // Material's diffuse color
//float3 g_vMaterialSpecular : Specular = float3( 1.0f, 1.0f, 0.5f );  // Material's specular color

////nice sun, blue sky
//float3 g_vMaterialAmbient : Ambient = float3( 0.87f, 0.87f, 0.87f );   // Material's ambient color
//float3 g_vMaterialDiffuse : Diffuse = float3( -0.15f, -0.18f, 0.1f );   // Material's diffuse color
//float3 g_vMaterialSpecular : Specular = float3( .2f, 0.2f, -0.2f );  // Material's specular color

//////ao only
//float3 g_vMaterialAmbient : Ambient = float3( 1.0f, 1.0f, 1.0f );   // Material's ambient color
//float3 g_vMaterialDiffuse : Diffuse = float3( 0.0f, 0.0f, 0.0f );   // Material's diffuse color
//float3 g_vMaterialSpecular : Specular = float3( 0.0f, 0.0f, 0.0f );  // Material's specular color

//synthetic gray theme
float3 g_vMaterialAmbient : Ambient = float3( 0.3f, 0.35f, 0.3f );   // Material's ambient color
float3 g_vMaterialDiffuse : Diffuse = float3( 0.9f, 0.9f, 1.0f );   // Material's diffuse color
float3 g_vMaterialSpecular : Specular = float3( 1.0f, 1.0f, 1.0f );  // Material's specular color


float  g_fMaterialAlpha : Opacity = 1.0f;
int    g_nMaterialShininess : SpecularPower = 128;
float  g_fMaterialSpecularSharpness : Sharpness = 1.0f;
float  g_fMaterialDiffuseGoochy : Goochy = 0.0f;
float  g_fMaterialDiffuseSharpness : Sharpness = 1.0f;

float3 g_vLightColor : LightColor = float3( 1.0f, 1.0f, 1.0f );        // Light color
float3 g_vLightPosition : LightPosition = float3( .71f, 0.0f, -.71f );   // Light position
float3 g_vCameraPosition : CameraPosition = float3( 1.0f, 1.0f, 1.0f );   // Camera position

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

   color.r=(2.0f*color.r)-1.0f;
   color.g=(2.0f*color.g)-1.0f;
   color.b=(2.0f*color.b)-1.0f;
   
   float4 light=normalize(float4(sin(Time)*0.5773f,abs(cos(Time*2.435)*0.5773f),-cos(Time*1.6435)*0.5773f,0));
   //float4 light=normalize(float4(.51,.51,.51,0));

   float ambient=sqrt(dot(color, color))*2.0;     //Ambient
   float diffuse=max(0,(dot(light, color)+g_fMaterialDiffuseGoochy/2.0f)*g_fMaterialDiffuseSharpness);    //Diffuse
   color=normalize(color);
   float4 halfvec=normalize(color+light);
   float blinn=pow(max(0,dot(halfvec, color)), g_nMaterialShininess);   //Blinn

   color.r=ambient*g_vMaterialAmbient.r + diffuse*g_vMaterialDiffuse.r + blinn*g_vMaterialSpecular.r*ambient;
   color.g=ambient*g_vMaterialAmbient.g + diffuse*g_vMaterialDiffuse.g + blinn*g_vMaterialSpecular.g*ambient;
   color.b=ambient*g_vMaterialAmbient.b + diffuse*g_vMaterialDiffuse.b + blinn*g_vMaterialSpecular.b*ambient;

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