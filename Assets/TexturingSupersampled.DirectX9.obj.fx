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
    MipFilter = LINEAR;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
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
                 out float4 vColorOut: TEXCOORD1,
                 uniform bool bSpecular
                )
{
	vPosProj = mul( vPosObject, WorldViewProjection );

	//to unmap uv

	vColorOut.rgb = vPosProj;
	vTexCoordOut = vTexCoordIn;
	vTexCoordOut.y = -vTexCoordOut.y;	
}



//--------------------------------------------------------------------------------------
// Name: Lighting
// Type: Pixel Shader
// Desc: Compute lighting and modulate the texture
//--------------------------------------------------------------------------------------
void Lighting( float2 vTexCoord: TEXCOORD0,
               float4 vColorIn: TEXCOORD1,
               out float4 vColorOut: COLOR0,
               uniform bool bTexture )
{  
    if(vColorIn.y > -0.52){              
    } else {
       const float aspectInv=9.0f/16.0f;
       vTexCoord.y /=aspectInv;
       vTexCoord.y -= .12;
    }



    // Sample and modulate the texture
    // float4 color=tex2D( MeshTextureSampler, vTexCoord ).rgba;

    const float offsetX = .5 / 1920.0 ;
    const float offsetY = .5 / 1080.0 ;

    int x=-1;
    int y=-1;
    
    float4 color = float4(0,0,0,0);
    float4 sample = float4(0,0,0,0);

    sample = tex2D( MeshTextureSampler, float2(vTexCoord.x+(offsetX*x), vTexCoord.y+(offsetY*y) ) ).rgba;
    color = float4( color.x + sample.x,
                        color.y + sample.y,
                        color.z + sample.z,
                        color.w + sample.w );
    
    x=0;
    y=-1;

    sample = tex2D( MeshTextureSampler, float2(vTexCoord.x+(offsetX*x), vTexCoord.y+(offsetY*y) ) ).rgba;
    color = float4( color.x + sample.x,
                        color.y + sample.y,
                        color.z + sample.z,
                        color.w + sample.w );
    
    x=1;
    y=-1;

    sample = tex2D( MeshTextureSampler, float2(vTexCoord.x+(offsetX*x), vTexCoord.y+(offsetY*y) ) ).rgba;
    color = float4( color.x + sample.x,
                        color.y + sample.y,
                        color.z + sample.z,
                        color.w + sample.w );
    
    x=-1;
    y=0;

    sample = tex2D( MeshTextureSampler, float2(vTexCoord.x+(offsetX*x), vTexCoord.y+(offsetY*y) ) ).rgba;
    color = float4( color.x + sample.x,
                        color.y + sample.y,
                        color.z + sample.z,
                        color.w + sample.w );
    
    x=0;
    y=0;

    sample = tex2D( MeshTextureSampler, float2(vTexCoord.x+(offsetX*x), vTexCoord.y+(offsetY*y) ) ).rgba;
    color = float4( color.x + sample.x,
                        color.y + sample.y,
                        color.z + sample.z,
                        color.w + sample.w );
    
    x=1;
    y=0;

    sample = tex2D( MeshTextureSampler, float2(vTexCoord.x+(offsetX*x), vTexCoord.y+(offsetY*y) ) ).rgba;
    color = float4( color.x + sample.x,
                        color.y + sample.y,
                        color.z + sample.z,
                        color.w + sample.w );
    
    x=-1;
    y=1;

    sample = tex2D( MeshTextureSampler, float2(vTexCoord.x+(offsetX*x), vTexCoord.y+(offsetY*y) ) ).rgba;
    color = float4( color.x + sample.x,
                        color.y + sample.y,
                        color.z + sample.z,
                        color.w + sample.w );

    x=0;
    y=1;

    sample = tex2D( MeshTextureSampler, float2(vTexCoord.x+(offsetX*x), vTexCoord.y+(offsetY*y) ) ).rgba;
    color = float4( color.x + sample.x,
                        color.y + sample.y,
                        color.z + sample.z,
                        color.w + sample.w );


    x=1;
    y=1;

    sample = tex2D( MeshTextureSampler, float2(vTexCoord.x+(offsetX*x), vTexCoord.y+(offsetY*y) ) ).rgba;
    color = float4( color.x + sample.x,
                        color.y + sample.y,
                        color.z + sample.z,
                        color.w + sample.w );
    
    
    
    
    color.x /= 9.0f;
    color.y /= 9.0f;
    color.z /= 9.0f;
    color.a = 1.0f;

    //if(vColorIn.y > -0.52){
    //  color.x = 1.0f;
    //  color.y = 0.0f;
    //  color.z = 0.0f;
    //} else {      
    //  color.x = 0.0f;
    //  color.y = 1.0f;
    //  color.z = 0.0f;
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
