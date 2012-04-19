//--------------------------------------------------------------------------------------
// File: Tutorial02.fx
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// Constant Buffer Variables
//--------------------------------------------------------------------------------------

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
    //float3 vLightPos = float3(0.71, 0.71,-.71);
    //float3 vLightPos = float3(0.0, 0.0, -1.0);
    float3 vLightPos = float3(5.0, 8.0, -5.0);
    float3 vEyePt = float3(5.0, 5.0, -5.0);
    float Explode = 0.01;
};

cbuffer cbChangesEveryFrame
{
    matrix World : World;
    matrix View : View;
    matrix Projection : Projection;
};

//--------------------------------------------------------------------------------------
// Structs
//--------------------------------------------------------------------------------------

struct VS_INPUT
{
    float3 Pos          : POSITION;         
    float3 Norm         : NORMAL;           
    float2 Tex          : TEXCOORD0; 
    float3 Tangent      : TANGENT;
};

struct GSPS_INPUT
{
    float4 Pos : SV_POSITION;
    float3 Norm : NORMAL;
    float2 Tex : TEXCOORD0;
    float3 Tangent : TANGENT;
};

//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------

GSPS_INPUT VS( VS_INPUT input )
{
    GSPS_INPUT output = (GSPS_INPUT)0;
    
    output.Tex = input.Tex;
    output.Pos = mul( mul( mul(float4(input.Pos.xyz, 1), World), View), Projection);
    
    output.Norm = normalize( mul( input.Norm, (float3x3)World) );
    output.Tangent = normalize( mul( input.Tangent, (float3x3)World) );
    
    return output;
}

//--------------------------------------------------------------------------------------
// Geometry Shader
//--------------------------------------------------------------------------------------
[maxvertexcount(12)]
void GS( triangle GSPS_INPUT input[3], inout TriangleStream<GSPS_INPUT> TriStream )
{
    GSPS_INPUT output;
    
    //
    // Calculate the face normal
    //
    float3 faceEdgeA = input[1].Pos - input[0].Pos;
    float3 faceEdgeB = input[2].Pos - input[0].Pos;
    float3 faceNormal = normalize( cross(faceEdgeA, faceEdgeB) );
    float3 ExplodeAmt = faceNormal*Explode;
    
    //
    // Calculate the face center
    //
    float3 centerPos = (input[0].Pos.xyz + input[1].Pos.xyz + input[2].Pos.xyz)/3.0;
    float2 centerTex = (input[0].Tex + input[1].Tex + input[2].Tex)/3.0;
    centerPos += ExplodeAmt;
    
    //
    // Output the pyramid
    //
    for( int i=0; i<3; i++ )
    {
        output.Pos = input[i].Pos + float4(ExplodeAmt,0);
        output.Pos = mul( output.Pos, View );
        output.Pos = mul( output.Pos, Projection );
        output.Norm = input[i].Norm;
        output.Tex = input[i].Tex;
        TriStream.Append( output );
        
      
        int iNext = (i+1)%3;
        output.Pos = input[iNext].Pos + float4(ExplodeAmt,0);
        output.Pos = mul( output.Pos, View );
        output.Pos = mul( output.Pos, Projection );
        output.Norm = input[iNext].Norm;
        output.Tex = input[iNext].Tex;
        TriStream.Append( output ); 
        
        output.Pos = float4(centerPos,1) + float4(ExplodeAmt,0);
        output.Pos = mul( output.Pos, View );
        output.Pos = mul( output.Pos, Projection );
        output.Norm = faceNormal;
        output.Tex = centerTex;
        TriStream.Append( output );
        
        TriStream.RestartStrip();
    }
    
    for( int i=2; i>=0; i-- )
    {
        output.Pos = input[i].Pos + float4(ExplodeAmt,0);
        output.Pos = mul( output.Pos, View );
        output.Pos = mul( output.Pos, Projection );
        output.Norm = -input[i].Norm;
        output.Tex = input[i].Tex;
        TriStream.Append( output );
    }
    TriStream.RestartStrip();
}


//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------
float4 PS( GSPS_INPUT input) : SV_Target
{
    //// Calculate lighting assuming light color is <1,1,1,1>
    
    float4 diffuse = g_txDiffuse.Sample( samLinear, input.Tex );
    float specMask = diffuse.a;
    float3 norm = g_txNormal.Sample( samLinear, input.Tex ).xyz;
    //norm *= 2;
    //norm -= float3(1,1,1);
    
    float3 binorm = normalize( cross( input.Norm, input.Tangent ) );
    float3x3 BTNMatrix = float3x3( binorm, input.Tangent, input.Norm );
    //norm = mul( norm, BTNMatrix );
    
    
    
    // Calculate specular power
    float3 viewDir = mul(input.Pos.xyz, (float3x3)World);
    float3 halfAngle = normalize( mul(vLightPos, (float3x3)View) - viewDir );
    float specPower = pow( saturate(dot( halfAngle, norm )), 16.0 );
    
    // diffuse lighting
    //float lighting = dot( normalize(vLightPos) , norm ) ;
    //lighting = (lighting+1.0)*0.25;
    
    //return lighting*diffuse+specPower*specMask*float4(1.0, 1.0, 1.0, 1.0);
    //return lighting*diffuse+specPower*float4(1.0, 1.0, 1.0, 1.0);
    //return specPower*float4(1.0, 1.0, 1.0, 1.0);
    return float4(norm.x,norm.y,norm.z, 1.0);
}


//--------------------------------------------------------------------------------------
// Shader Techniques and Passes
//--------------------------------------------------------------------------------------
technique Render9
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_3_0, VS() ) );
        SetPixelShader( CompileShader( ps_3_0, PS() ) );
    }
}

technique10 Render
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_4_0, VS() ) );
        //SetGeometryShader( CompileShader( gs_4_0, GS() ) );
	    SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, PS() ) );
    }
}
