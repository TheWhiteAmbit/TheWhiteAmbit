using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Interop;
using System.Runtime.InteropServices;
using TheWhiteAmbit.ClrWrapperLib;

namespace WPFApp
{
    /// <summary>
    /// Interaction logic for Window1.xaml
    /// </summary>
    public partial class Window1 : Window
    {
        private int RESOLUTION_X = 1280;
        private int RESOLUTION_Y = 720;
        private double Z_NEAR = 0.01;

        private readonly D3DImage d3DImage;
        private IntPtr sceneIntPtr;

        public Window1()
        {   
            // parse the XAML
            InitializeComponent();
 
            // Create a D3DImage to host the scene and
            // monitor it for changes in front buffer availability
            d3DImage = new D3DImage();
            d3DImage.IsFrontBufferAvailableChanged += OnIsFrontBufferAvailableChanged;

            // begin rendering the custom D3D scene into the D3DImage
            InitRenderingScene();
            //ImageBrush imageBrush = new ImageBrush(d3dImage);
            //ImageSource imageSource = d3dImage;
            directXImage.Source = d3DImage;
            //Background = new ImageBrush(d3dImage);
            //Foreground = new ImageBrush(d3dImage);
            //directXImage.OpacityMask = imageBrush.ImageSource;
            //directXImage2.Source = imageSource;
            //mainGrid.Background = new ImageBrush(d3dImage);
        }

        private void OnIsFrontBufferAvailableChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            // if the front buffer is available, then WPF has just Created a new
            // D3D device, so we need to start rendering our custom scene
            if (d3DImage.IsFrontBufferAvailable)
            {
                InitRenderingScene();
            }
            else
            {
                // If the front buffer is no longer available, then WPF has lost its
                // D3D device so there is no reason to waste cycles rendering our
                // custom scene until a new device is Created.
                StopRenderingScene();
            }
        }

        double timerValue;

        SceneGraphFactory factory;
        CameraAsset cameraAsset;
        TransformNode drawmockTransformNodeFirst;
        TransformNode drawmockTransformNodeSecond;
        DrawMockNode drawMockNode;
        BaseNode clearTargetBuffer;
        BaseNode clearLayerBuffer;
        TargetRenderAsset wpfRenderer;
        TargetRenderAsset targetRenderer;
        CudaRaytraceRenderAsset cudaRenderer;
        EffectAsset effectAssetLayer;
        EffectAsset effectAssetMesh;

        private void InitRenderingScene()
        {
            if (!d3DImage.IsFrontBufferAvailable)
                return;
            factory = new SceneGraphFactory();      
            wpfRenderer = factory.CreateRenderTargetA8R8G8B8Asset();
		    	
            /////////////////////////////////////////////////////////////

            cameraAsset = factory.CreateCamera();
            cameraAsset.Perspective(34.0 , (double)RESOLUTION_X / (double)RESOLUTION_Y, Z_NEAR, 0.0);
        

            targetRenderer = factory.CreateRenderTargetAsset();            

            clearTargetBuffer = factory.CreateClearBuffer();
            targetRenderer.SetRoot(clearTargetBuffer);
            clearLayerBuffer = factory.CreateClearBuffer();
            wpfRenderer.SetRoot(clearLayerBuffer);

            TextureLayerNode targetTextureLayer = factory.CreateTextureLayer(targetRenderer);            
            targetTextureLayer.MakeParent(clearLayerBuffer);
                
            effectAssetMesh = factory.CreateEffect("Position.DirectX9.obj.fx");
            effectAssetMesh.SetValue("Time", 1);

            ObjMeshAsset objMeshAsset = factory.CreateObjMesh("bigguy_g.obj");
            drawMockNode = factory.CreateDrawMock();            
            drawMockNode.SetMesh(objMeshAsset);
            drawmockTransformNodeFirst = factory.CreateTransform();
            drawmockTransformNodeSecond = factory.CreateTransform();
            drawmockTransformNodeFirst.MakeParent(clearTargetBuffer);
            drawmockTransformNodeSecond.MakeParent(drawmockTransformNodeFirst);
            drawMockNode.MakeParent(drawmockTransformNodeSecond);

            ///////////////////////////////////////////////

            cudaRenderer = factory.CreateCudaRaytraceRenderAsset();
            TextureLayerNode cudaTextureLayer = factory.CreateTextureLayer(cudaRenderer);
            cudaRenderer.SetRoot(drawMockNode);                        
            cudaTextureLayer.MakeParent(clearLayerBuffer);
            cudaRenderer.SetTextureTarget(4, factory.CreateTextureAsset(targetRenderer));

            ////////////////////////////////////

            effectAssetLayer = factory.CreateEffect("TextureLayerWpf.DirectX9.fx");

            // set the back buffer using the new scene pointer
            sceneIntPtr = (IntPtr)wpfRenderer.GetDirect3D9Surface(0);
            d3DImage.Lock();
            d3DImage.SetBackBuffer(D3DResourceType.IDirect3DSurface9, sceneIntPtr);
            d3DImage.Unlock();

            // leverage the Rendering event of WPF's composition target to
            // update the custom D3D scene
            CompositionTarget.Rendering += OnRendering;
        }

        private void StopRenderingScene()
        {
            // This method is called when WPF loses its D3D device.
            // In such a circumstance, it is very likely that we have lost 
            // our custom D3D device also, so we should just release the scene.
            // We will Create a new scene when a D3D device becomes 
            // available again.
            CompositionTarget.Rendering -= OnRendering;
            sceneIntPtr = IntPtr.Zero;
        }

        private void OnRendering(object sender, EventArgs e)
        {
            // when WPF's composition target is about to render, we update our 
            // custom render target so that it can be blended with the WPF target
            UpdateScene();
        }

        private void UpdateScene()
        {
            if (!d3DImage.IsFrontBufferAvailable || sceneIntPtr == IntPtr.Zero)
                return;
            timerValue += .02;
            const double distance = 4;
            //cameraAsset.LookAt(
            //    -distance * Math.Cos(timerValue), 
            //    //1.4 * Math.Sin(timerValue * 2.1), 
            //    2.0,
            //    distance * Math.Sin(timerValue),
            //    0, 0, 0,
            //    0, 1, 0);

            cameraAsset.LookAt(distance * Math.Cos(timerValue) + 1.5, 1.0, distance * Math.Sin(timerValue) + 1.5,
                             0.00123789, .1823748590456, 0.001435789, 0.01234, 0.9934256, 0.00964578);

            clearTargetBuffer.ApplyCamera(cameraAsset);
                
            targetRenderer.Present(effectAssetMesh);
            cudaRenderer.Present(0);
            // lock the D3DImage
            d3DImage.Lock();
            wpfRenderer.Present(effectAssetLayer);            
            // invalidate the updated region of the D3DImage (in this case, the whole image)
            d3DImage.AddDirtyRect(new Int32Rect(0, 0, RESOLUTION_X, RESOLUTION_Y));
            // unlock the D3DImage
            d3DImage.Unlock();
        }

        private void Button_Click(object sender, RoutedEventArgs e) {
            StopRenderingScene();
            Environment.Exit(0); //causes error
        }
    }
}
