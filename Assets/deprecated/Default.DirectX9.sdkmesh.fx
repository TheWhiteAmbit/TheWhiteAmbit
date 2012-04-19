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
float    g_fTime : Time;            // App's time in seconds
float4x4 World : World;          // World matrix
float4x4 View : View;          // World matrix
float4x4 WorldViewProjection : WorldViewProjection; // World * View * Projection matrix


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
    // for display
    vPosProj = mul( vPosObject, WorldViewProjection );
    
    // Transform the normal into world space for lighting
    float3 vNormalWorld = vNormalObject;

    // Pass the texture coordinate
    vTexCoordOut = vTexCoordIn;
    
    // Compute the ambient and diffuse components of illumination        
    vColorOut = float4(vNormalWorld.x*.5+.5, vNormalWorld.y*.5+.5, vNormalWorld.z*.5+.5, 1.0f/vPosProj.z);
    //vColorOut = float4(vPosProj.x*.5+.5, vPosProj.y*.5+.5, vPosProj.z*.5+.5, 1.0f);
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
    vColorOut = vColorIn; 
}


//--------------------------------------------------------------------------------------
// Techniques
//--------------------------------------------------------------------------------------
technique Render9
{
    pass P0
    {
        VertexShader = compile vs_2_0 Projection(true);    
        PixelShader = compile ps_2_0 Lighting(false);    
    }
}
