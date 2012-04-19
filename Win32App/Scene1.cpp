
#include "Scene1.h"
#include "../DirectX9Lib/DirectX9DrawMock.h"
#include "../DirectX9Lib/DirectX9ClearBuffer.h"
#include "../DirectX9Lib/DirectX9TextureLayer.h"
#include "../SceneGraphLib/Camera.h"

#include "../CudaLib/rendering.h"

#include <math.h>

using namespace TheWhiteAmbit;

Scene1::Scene1(HWND a_hWnd)
{
	renderer=new DirectX9Renderer(a_hWnd);
}

Scene1::~Scene1(void)
{
}

unsigned int Scene1::run(void)
{
	double number=0.0f;
	
	DirectX9ViewRender* viewRender=new DirectX9ViewRender(renderer);
	DirectX9DrawMock* mockNode=new DirectX9DrawMock(renderer);
	Node* rootNode=new DirectX9ClearBuffer(renderer);
	
	DirectX9TargetRender* targetRender=new DirectX9TargetRender(renderer);	
	DirectX9TargetRender* targetRender1=new DirectX9TargetRender(renderer);	
	DirectX9TextureLayer* layerNode=new DirectX9TextureLayer(renderer);

	viewRender->setRootNode(rootNode);
	layerNode->makeParent(rootNode);

	DirectX9ClearBuffer* targetClsNode=new DirectX9ClearBuffer(renderer);
	targetRender->setRootNode(targetClsNode);
	targetRender1->setRootNode(targetClsNode);
	mockNode->makeParent(targetClsNode);

	//mockNode->makeParent(rootNode);
	//layerNode->makeParent(rootNode);

	DirectX9Effect* positionEffect=new DirectX9Effect(renderer, L"Position.DirectX9.obj.fx");	
	DirectX9Effect* trackingEffect=new DirectX9Effect(renderer, L"Tracking.DirectX9.obj.fx");	
	DirectX9Effect* layerEffect0=new DirectX9Effect(renderer, L"TextureLayer.DirectX9.fx");
	DirectX9Effect* layerEffect1=new DirectX9Effect(renderer, L"TextureLayer.Deferred.Lighting.DirectX9.fx");

	positionEffect->setValue(L"Time", 1.0);
	trackingEffect->setValue(L"Time", 1.0);
	layerEffect0->setValue(L"Time", 1.0);
	layerEffect1->setValue(L"Time", 1.0);
	
	//DirectX9ObjMesh* mesh=new DirectX9ObjMesh(renderer, L"tumbler.obj");
	DirectX9ObjMesh* mesh=new DirectX9ObjMesh(renderer, L"bigguy_g.obj");
	//DirectX9ObjMesh* mesh=new DirectX9ObjMesh(renderer, L"thunderchild.obj");
	//DirectX9ObjMesh* mesh=new DirectX9ObjMesh(renderer, L"imrod400k.obj");
	mockNode->setMesh(mesh);
	
	CudaRaytraceRender* raytracer=new CudaRaytraceRender(renderer);
	DirectX9Texture* raytraceTexture=new DirectX9Texture(renderer);
		
	raytracer->setRootNode(mockNode);	
	raytracer->setTextureTarget(4, targetRender->getTexture());
	raytracer->setTextureTarget(5, targetRender1->getTexture());

	raytracer->setTextureTarget(0, raytraceTexture);

	layerNode->setTextureSource(0, raytraceTexture);
	double dist=abs(0.1*tan(number*0.3))+3.0;
	Camera camera;
	
	while(true)
	{
		number+=0.001;
		camera.perspective(35.0, RESOLUTION_X/RESOLUTION_Y, 0.03125, 0.0);
		double dist=abs(0.1*tan(number*0.3))+3.0;
		camera.lookAt(
			D3DXVECTOR3(dist*-cos(number), 1.8f, dist*sin(number)), 
			D3DXVECTOR3(0,0,0), 
			D3DXVECTOR3(0,1,0));
		TransformVisitor transformNodeVisitor;
		camera.acceptEnter(&transformNodeVisitor);
		mockNode->acceptEnter(&transformNodeVisitor);

		layerEffect0->setValue(L"Time", number*5.0);
		layerEffect1->setValue(L"Time", number*5.0);
		
		targetRender1->present(trackingEffect);
		targetRender->present(positionEffect);
		raytracer->present(21);
		viewRender->present(layerEffect1);

		MementoVisitor mementoNodeVisitor;
		mockNode->accept(&mementoNodeVisitor);
		Sleep(0);
	}
	return 0;
}
