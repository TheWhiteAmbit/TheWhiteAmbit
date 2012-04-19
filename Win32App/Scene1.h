#pragma once
#include "../win32lib/irunnable.h"
#include "../SceneGraphLib/Node.h"
#include "../DirectX9Lib/DirectX9ViewRender.h"
#include "../DirectX9Lib/DirectX9Effect.h"
#include "../DirectX9Lib/DirectX9DrawMock.h"
#include "../DirectX9Lib/DirectX9TargetRender.h"
#include "../CudaLib/CudaRaytraceRender.h"


class Scene1 :
	public IRunnable
{
 	TheWhiteAmbit::DirectX9Renderer*	renderer;
public:
	Scene1(HWND);
	virtual ~Scene1(void);
	virtual unsigned int run(void);
};
