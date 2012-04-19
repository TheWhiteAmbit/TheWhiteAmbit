//---------------------------------------------------------------------
// File: Tutorial02.fx
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//---------------------------------------------------------------------

//---------------------------------------------------------------------
// Constant Buffer Variables
//---------------------------------------------------------------------

Texture2D g_txDiffuse;
Texture2D g_txNormal;
Texture2D g_txSpecular;

SamplerState samLinear
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

cbuffer cbConstant
{
    float3 vLightDir = float3(-0.71,0.71,-.71);
    //float3 vLightDir = float3(0.0,0.0,-1.0);
    float Explode = 0.01;
};

cbuffer cbChangesEveryFrame
{
    matrix World : World;
    matrix View : View;
    matrix Projection : Projection;
};

//---------------------------------------------------------------------
// Structs
//---------------------------------------------------------------------

struct VS_INPUT
{
    float4 Pos          : POSITION;         
    float3 Norm         : NORMAL;           
    float2 Tex          : TEXCOORD0;        
};

struct GSPS_INPUT
{
    float4 Pos : SV_POSITION;
    float3 Norm : NORMAL;
    float2 Tex : TEXCOORD0;
};

//---------------------------------------------------------------------
// Vertex Shader
//---------------------------------------------------------------------

GSPS_INPUT VS( VS_INPUT input )
{
    GSPS_INPUT output = (GSPS_INPUT)0;
    
    output.Tex = input.Tex;
    output.Pos = mul( mul( mul(float4(input.Pos.xyz, 1), World), View), Projection);
    
    output.Norm = mul( mul( input.Norm, (float3x3)World ), (float3x3)View);
    //output.Norm = input.Norm;
    output.Norm = normalize(output.Norm);
    
    return output;
}

//---------------------------------------------------------------------
// Geometry Shader
//---------------------------------------------------------------------

[maxvertexcount(3)]
void GS( triangle GSPS_INPUT input[3], inout TriangleStream<GSPS_INPUT> TriStream )
{
    GSPS_INPUT output;
    
    for( int i=0; i<3; i++ )
    {
        output.Pos = input[i].Pos;
        output.Pos = mul( output.Pos, World );
        output.Pos = mul( output.Pos, View );
        output.Pos = mul( output.Pos, Projection );
        output.Norm = input[i].Norm;
        output.Tex = input[i].Tex;
        TriStream.Append( output );
    }
    TriStream.RestartStrip();
}


//---------------------------------------------------------------------
// Pixel Shader
//---------------------------------------------------------------------

float4 PS( GSPS_INPUT input) : SV_Target
{
    //// Calculate lighting assuming light color is <1,1,1,1>
    
    return float4(1.0, 1.0, 0.0, 1.0);
}


//---------------------------------------------------------------------
// Shader Techniques and Passes
//---------------------------------------------------------------------

technique10 Render
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_4_0, VS() ) );
	  SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, PS() ) );
    }
}
