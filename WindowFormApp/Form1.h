//The White Ambit, Rendering-Framework
//Copyright (C) 2009  Moritz Strickhausen

//This program is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with this program.  If not, see <http://www.gnu.org/licenses/>.



#pragma once
#using "ClrWrapperLib.dll"

namespace WindowFormApp {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace TheWhiteAmbit::ClrWrapperLib;

	/// <summary>
	/// Summary for Form1
	///
	/// WARNING: If you change the name of this class, you will need to change the
	///          'Resource File Name' property for the managed resource compiler tool
	///          associated with all .resx files this class depends on.  Otherwise,
	///          the designers will not be able to interact properly with localized
	///          resources associated with this form.
	/// </summary>
	public ref class Form1 : public System::Windows::Forms::Form
	{
	private: System::Windows::Forms::ToolStripMenuItem^  deferedLightingToolStripMenuItem;
	private: System::Windows::Forms::Timer^  timer1;
	private: System::Windows::Forms::ToolStripMenuItem^  combineToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  videoToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  saveScreenshotToolStripMenuItem;
	private: System::Windows::Forms::SaveFileDialog^  saveFileDialog1;
	private: System::Windows::Forms::CheckBox^  checkBox1;
	public:
		Form1(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//

			this->factory= gcnew SceneGraphFactory(this->panel1->Handle);

			this->viewLayerRenderer = factory->CreateView();
			this->targetRenderer = factory->CreateRenderTargetAsset();
			this->targetRendererMemento0 = factory->CreateRenderTargetAsset();
			this->targetRendererMemento1 = factory->CreateRenderTargetAsset();
			this->targetRendererMemento2 = factory->CreateRenderTargetAsset();
			this->targetRendererMemento3 = factory->CreateRenderTargetAsset();
			this->cudaRenderer = factory->CreateCudaRaytraceRenderAsset();

			this->cudaVideoRenderer = factory->CreateCudaVideoRenderAsset("default.m4v");
			//this->cudaFileRenderer = factory->CreateCudaFileRenderAsset("encode.m4v");			
			if(cudaFileRenderer!=nullptr)
				cudaFileRenderer->SetTextureSource(0, factory->CreateTextureAsset(targetRenderer));
						

			this->effectAssetTarget = factory->CreateEffect("Position.DirectX9.obj.fx");
			//this->effectAssetTarget = factory->CreateEffect("Texturing.DirectX9.obj.fx");
			//this->effectAssetTarget = factory->CreateEffect("LedXYOffset16.DirectX9.obj.fx");
			this->effectAssetMemento = factory->CreateEffect("Tracking.DirectX9.obj.fx");	
			this->effectAssetMemento1 = factory->CreateEffect("Planes.DirectX9.obj.fx");	
			this->effectAssetMemento2 = factory->CreateEffect("Backplanes.DirectX9.obj.fx");	
			this->effectAssetViewLayer = factory->CreateEffect("TextureLayer.DirectX9.fx");

			//this->effectAssetTarget->SetValue("g_MeshTexture", factory->CreateTextureAsset(gcnew Bitmap("texturelayout.png")));
			//this->effectAssetTarget->SetValue("g_MeshTexture", factory->CreateTextureAsset(cudaVideoRenderer));
			//this->effectAssetTarget->SetValue("g_MeshTexture", factory->CreateTextureAsset(gcnew Bitmap("texturelayout.png")));
			if(cudaVideoRenderer!=nullptr)
				this->effectAssetTarget->SetValue("g_MeshTexture", factory->CreateTextureAsset(cudaVideoRenderer));

			this->cameraAsset = factory->CreateCamera();
			//this->cameraAsset->Perspective(74.0, (double)RESOLUTION_X/(double)RESOLUTION_Y, 0.01, 0.0);
			//this->cameraAsset->Orthogonal(128.0, 72.0, 0.1, 100.0);
			//this->cameraAsset->LookAt((10-(RESOLUTION_X/2.0))*.03, 10, (10-(RESOLUTION_Y/2.0))*.03,	0,0,0,	0,1,0);
			//this->cameraAsset->LookAt(64, 36, 10,	64, 36, 0,	0,1,0);

			this->targetRoot = factory->CreateClearBuffer();

			this->transformNode = factory->CreateTransform();
			this->transformNode->MakeParent(targetRoot);

			this->transformNode2 = factory->CreateTransform();
			this->transformNode2->MakeParent(transformNode);

			//DrawMockNode^ drawMockNode = factory->CreateDrawMock();
			//EffectAsset^ effectAsset = factory->CreateEffect("deprecated/test2.fx");
			//drawMockNode->setEffect(effectAsset);
			//drawMockNode->setParent(transformNode2);
			//SdkMeshAsset^ SdkMeshAsset = factory->CreateSdkMesh("deprecated/ut3.sdkmesh");
			//drawMockNode->setMesh(SdkMeshAsset);

			this->treeView2->Nodes->Add("Effects");
			this->treeView2->Nodes->Add("Geometrie");
			this->treeView2->Nodes->Add("Lights");
			this->treeView2->Nodes->Add("Cameras");
			this->treeView2->Nodes->Add("Textures");

			this->targetRenderer->SetRoot(targetRoot);
			this->targetRendererMemento0->SetRoot(targetRoot);
			this->targetRendererMemento1->SetRoot(targetRoot);
			this->targetRendererMemento2->SetRoot(targetRoot);
			this->targetRendererMemento3->SetRoot(targetRoot);

			this->layerRoot = factory->CreateClearBuffer();
			this->viewLayerRenderer->SetRoot(layerRoot);
			//this->viewLayerRenderer->SetRoot(targetRoot);

			this->panel1->Paint += gcnew System::Windows::Forms::PaintEventHandler(this, &Form1::Panel1_OnPaint);
			this->panel1->MouseClick += gcnew System::Windows::Forms::MouseEventHandler(this, &Form1::Panel1_OnMouseClick);
			this->panel1->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &Form1::Panel1_OnMouseMove);

			this->treeView1->Nodes->Add(
				this->BaseNode_getTreeNode(targetRoot, gcnew TreeNode("Scene")));

			this->treeView3->Nodes->Add(
				this->BaseNode_getTreeNode(layerRoot, gcnew TreeNode("Layer")));

		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~Form1()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Panel^  panel1;
	private: System::Windows::Forms::TrackBar^  trackBar1;
	private: System::Windows::Forms::MenuStrip^  menuStrip1;
	private: System::Windows::Forms::ToolStripMenuItem^  toolStripMenuItem1;
	private: System::Windows::Forms::ToolStripMenuItem^  openToolStripMenuItem;
	private: System::Windows::Forms::TreeView^  treeView1;
	private: System::Windows::Forms::StatusStrip^  statusStrip1;
	private: System::Windows::Forms::OpenFileDialog^  openFileDialog1;

	private: System::Windows::Forms::TreeView^  treeView2;
	private: System::Windows::Forms::OpenFileDialog^  openFileDialog2;
	private: System::Windows::Forms::ToolStripStatusLabel^  toolStripStatusLabel1;


	private: System::Windows::Forms::ToolStripMenuItem^  taskToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  raytraceCPUToolStripMenuItem;

	private: System::Windows::Forms::ToolStripMenuItem^  raytraceGPUToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  renderTargetToolStripMenuItem;
	private: System::Windows::Forms::TreeView^  treeView3;
	private: System::Windows::Forms::ToolStripMenuItem^  measureToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  xGPUToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  xDXToolStripMenuItem;
	private: System::Windows::Forms::Button^  button1;
	private: System::ComponentModel::BackgroundWorker^  backgroundWorker1;
	private: System::Windows::Forms::NumericUpDown^  numericUpDown1;
	private: System::ComponentModel::IContainer^  components;






	protected: 

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>


#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->components = (gcnew System::ComponentModel::Container());
			this->panel1 = (gcnew System::Windows::Forms::Panel());
			this->trackBar1 = (gcnew System::Windows::Forms::TrackBar());
			this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			this->toolStripMenuItem1 = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->openToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->taskToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->renderTargetToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->raytraceCPUToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->raytraceGPUToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->deferedLightingToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->combineToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->videoToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->measureToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->xGPUToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->xDXToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->treeView1 = (gcnew System::Windows::Forms::TreeView());
			this->statusStrip1 = (gcnew System::Windows::Forms::StatusStrip());
			this->toolStripStatusLabel1 = (gcnew System::Windows::Forms::ToolStripStatusLabel());
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->treeView2 = (gcnew System::Windows::Forms::TreeView());
			this->openFileDialog2 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->treeView3 = (gcnew System::Windows::Forms::TreeView());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->backgroundWorker1 = (gcnew System::ComponentModel::BackgroundWorker());
			this->numericUpDown1 = (gcnew System::Windows::Forms::NumericUpDown());
			this->timer1 = (gcnew System::Windows::Forms::Timer(this->components));
			this->checkBox1 = (gcnew System::Windows::Forms::CheckBox());
			this->saveScreenshotToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->saveFileDialog1 = (gcnew System::Windows::Forms::SaveFileDialog());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->trackBar1))->BeginInit();
			this->menuStrip1->SuspendLayout();
			this->statusStrip1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->numericUpDown1))->BeginInit();
			this->SuspendLayout();
			this->panel1->BackColor = System::Drawing::Color::Transparent;
			this->panel1->Cursor = System::Windows::Forms::Cursors::NoMove2D;
			this->panel1->ForeColor = System::Drawing::SystemColors::WindowText;
			this->panel1->Location = System::Drawing::Point(12, 27);
			this->panel1->MaximumSize = System::Drawing::Size(1280, 720);
			this->panel1->MinimumSize = System::Drawing::Size(1280, 720);
			this->panel1->Name = L"panel1";
			this->panel1->Size = System::Drawing::Size(1280, 720);
			this->panel1->TabIndex = 0;
			this->trackBar1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Left) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->trackBar1->LargeChange = 10;
			this->trackBar1->Location = System::Drawing::Point(12, 768);
			this->trackBar1->Maximum = 1000;
			this->trackBar1->Name = L"trackBar1";
			this->trackBar1->Size = System::Drawing::Size(1214, 45);
			this->trackBar1->TabIndex = 1;
			this->trackBar1->Scroll += gcnew System::EventHandler(this, &Form1::trackBar1_Scroll);
			this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {this->toolStripMenuItem1, this->taskToolStripMenuItem, 
				this->measureToolStripMenuItem});
			this->menuStrip1->Location = System::Drawing::Point(0, 0);
			this->menuStrip1->Name = L"menuStrip1";
			this->menuStrip1->Size = System::Drawing::Size(1429, 24);
			this->menuStrip1->TabIndex = 2;
			this->menuStrip1->Text = L"menuStrip1";
			this->toolStripMenuItem1->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {this->openToolStripMenuItem, 
				this->saveScreenshotToolStripMenuItem});
			this->toolStripMenuItem1->Name = L"toolStripMenuItem1";
			this->toolStripMenuItem1->Size = System::Drawing::Size(37, 20);
			this->toolStripMenuItem1->Text = L"File";
			this->openToolStripMenuItem->Name = L"openToolStripMenuItem";
			this->openToolStripMenuItem->Size = System::Drawing::Size(168, 22);
			this->openToolStripMenuItem->Text = L"Open...";
			this->openToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::openToolStripMenuItem_Click);
			this->taskToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(6) {this->renderTargetToolStripMenuItem, 
				this->raytraceCPUToolStripMenuItem, this->raytraceGPUToolStripMenuItem, this->deferedLightingToolStripMenuItem, this->combineToolStripMenuItem, 
				this->videoToolStripMenuItem});
			this->taskToolStripMenuItem->Name = L"taskToolStripMenuItem";
			this->taskToolStripMenuItem->Size = System::Drawing::Size(50, 20);
			this->taskToolStripMenuItem->Text = L" Layer";
			this->renderTargetToolStripMenuItem->Name = L"renderTargetToolStripMenuItem";
			this->renderTargetToolStripMenuItem->Size = System::Drawing::Size(162, 22);
			this->renderTargetToolStripMenuItem->Text = L"RenderTarget";
			this->renderTargetToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::renderTargetToolStripMenuItem_Click);
			this->raytraceCPUToolStripMenuItem->Name = L"raytraceCPUToolStripMenuItem";
			this->raytraceCPUToolStripMenuItem->Size = System::Drawing::Size(162, 22);
			this->raytraceCPUToolStripMenuItem->Text = L"Raytrace CPU";
			this->raytraceCPUToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::raytraceCPUToolStripMenuItem_Click);
			this->raytraceGPUToolStripMenuItem->Name = L"raytraceGPUToolStripMenuItem";
			this->raytraceGPUToolStripMenuItem->Size = System::Drawing::Size(162, 22);
			this->raytraceGPUToolStripMenuItem->Text = L"Raytrace GPU";
			this->raytraceGPUToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::raytraceGPUToolStripMenuItem_Click);
			this->deferedLightingToolStripMenuItem->Name = L"deferedLightingToolStripMenuItem";
			this->deferedLightingToolStripMenuItem->Size = System::Drawing::Size(162, 22);
			this->deferedLightingToolStripMenuItem->Text = L"Defered Lighting";
			this->deferedLightingToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::deferedLightingToolStripMenuItem_Click);
			this->combineToolStripMenuItem->Name = L"combineToolStripMenuItem";
			this->combineToolStripMenuItem->Size = System::Drawing::Size(162, 22);
			this->combineToolStripMenuItem->Text = L"Combine";
			this->combineToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::combineToolStripMenuItem_Click);
			this->videoToolStripMenuItem->Name = L"videoToolStripMenuItem";
			this->videoToolStripMenuItem->Size = System::Drawing::Size(162, 22);
			this->videoToolStripMenuItem->Text = L"Video";
			this->videoToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::videoToolStripMenuItem_Click);
			this->measureToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {this->xGPUToolStripMenuItem, 
				this->xDXToolStripMenuItem});
			this->measureToolStripMenuItem->Name = L"measureToolStripMenuItem";
			this->measureToolStripMenuItem->Size = System::Drawing::Size(64, 20);
			this->measureToolStripMenuItem->Text = L"Measure";
			this->xGPUToolStripMenuItem->Name = L"xGPUToolStripMenuItem";
			this->xGPUToolStripMenuItem->Size = System::Drawing::Size(138, 22);
			this->xGPUToolStripMenuItem->Text = L"100x CUDA";
			this->xGPUToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::xGPUToolStripMenuItem_Click);
			this->xDXToolStripMenuItem->Name = L"xDXToolStripMenuItem";
			this->xDXToolStripMenuItem->Size = System::Drawing::Size(138, 22);
			this->xDXToolStripMenuItem->Text = L"100x DirectX";
			this->xDXToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::xDXToolStripMenuItem_Click);
			this->treeView1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>(((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom) 
				| System::Windows::Forms::AnchorStyles::Right));
			this->treeView1->Location = System::Drawing::Point(1313, 249);
			this->treeView1->Name = L"treeView1";
			this->treeView1->Size = System::Drawing::Size(104, 308);
			this->treeView1->TabIndex = 3;
			this->treeView1->DragDrop += gcnew System::Windows::Forms::DragEventHandler(this, &Form1::treeView1_DragDrop);
			this->treeView1->AfterSelect += gcnew System::Windows::Forms::TreeViewEventHandler(this, &Form1::treeView1_AfterSelect);
			this->treeView1->DragEnter += gcnew System::Windows::Forms::DragEventHandler(this, &Form1::treeView1_DragDrop);
			this->treeView1->ItemDrag += gcnew System::Windows::Forms::ItemDragEventHandler(this, &Form1::treeView1_ItemDrag);
			this->statusStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) {this->toolStripStatusLabel1});
			this->statusStrip1->Location = System::Drawing::Point(0, 819);
			this->statusStrip1->Name = L"statusStrip1";
			this->statusStrip1->Size = System::Drawing::Size(1429, 22);
			this->statusStrip1->TabIndex = 4;
			this->statusStrip1->Text = L"statusStrip1";
			this->toolStripStatusLabel1->Name = L"toolStripStatusLabel1";
			this->toolStripStatusLabel1->Size = System::Drawing::Size(118, 17);
			this->toolStripStatusLabel1->Text = L"toolStripStatusLabel1";
			this->openFileDialog1->FileName = L"openFileDialog1";
			this->treeView2->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->treeView2->Location = System::Drawing::Point(1313, 563);
			this->treeView2->Name = L"treeView2";
			this->treeView2->Size = System::Drawing::Size(104, 199);
			this->treeView2->TabIndex = 6;
			this->openFileDialog2->FileName = L"openFileDialog2";
			this->treeView3->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Right));
			this->treeView3->Location = System::Drawing::Point(1313, 27);
			this->treeView3->Name = L"treeView3";
			this->treeView3->Size = System::Drawing::Size(104, 216);
			this->treeView3->TabIndex = 7;
			this->button1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->button1->Location = System::Drawing::Point(1342, 767);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(75, 23);
			this->button1->TabIndex = 8;
			this->button1->Text = L"Raytrace Efx";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &Form1::button1_Click);
			this->backgroundWorker1->DoWork += gcnew System::ComponentModel::DoWorkEventHandler(this, &Form1::backgroundWorker1_DoWork);
			this->numericUpDown1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->numericUpDown1->Location = System::Drawing::Point(1342, 796);
			this->numericUpDown1->Name = L"numericUpDown1";
			this->numericUpDown1->Size = System::Drawing::Size(75, 20);
			this->numericUpDown1->TabIndex = 9;
			this->timer1->Tick += gcnew System::EventHandler(this, &Form1::timer1_Tick);
			this->checkBox1->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->checkBox1->AutoSize = true;
			this->checkBox1->Checked = true;
			this->checkBox1->CheckState = System::Windows::Forms::CheckState::Checked;
			this->checkBox1->Location = System::Drawing::Point(1267, 773);
			this->checkBox1->Name = L"checkBox1";
			this->checkBox1->Size = System::Drawing::Size(69, 17);
			this->checkBox1->TabIndex = 10;
			this->checkBox1->Text = L"memento";
			this->checkBox1->UseVisualStyleBackColor = true;
			this->checkBox1->CheckedChanged += gcnew System::EventHandler(this, &Form1::checkBox1_CheckedChanged);
			this->saveScreenshotToolStripMenuItem->Name = L"saveScreenshotToolStripMenuItem";
			this->saveScreenshotToolStripMenuItem->Size = System::Drawing::Size(168, 22);
			this->saveScreenshotToolStripMenuItem->Text = L"Save Screenshot...";
			this->saveScreenshotToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::saveScreenshotToolStripMenuItem_Click);
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1429, 841);
			this->Controls->Add(this->checkBox1);
			this->Controls->Add(this->treeView2);
			this->Controls->Add(this->treeView1);
			this->Controls->Add(this->treeView3);
			this->Controls->Add(this->numericUpDown1);
			this->Controls->Add(this->button1);
			this->Controls->Add(this->trackBar1);
			this->Controls->Add(this->panel1);
			this->Controls->Add(this->statusStrip1);
			this->Controls->Add(this->menuStrip1);
			this->MainMenuStrip = this->menuStrip1;
			this->Name = L"Form1";
			this->SizeGripStyle = System::Windows::Forms::SizeGripStyle::Show;
			this->Text = L"Form1";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->trackBar1))->EndInit();
			this->menuStrip1->ResumeLayout(false);
			this->menuStrip1->PerformLayout();
			this->statusStrip1->ResumeLayout(false);
			this->statusStrip1->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^  >(this->numericUpDown1))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
		static int RESOLUTION_X=1280;
		static int RESOLUTION_Y=720;
		CameraAsset^ cameraAsset;
		TransformNode^ transformNode;
		TransformNode^ transformNode2;
		DrawMockNode^ drawMockNode;
		BaseNode^	targetRoot;
		BaseNode^	layerRoot;
		ViewRenderAsset^ viewLayerRenderer;		
		TargetRenderAsset^ targetRenderer;
		TargetRenderAsset^ targetRendererMemento0;
		TargetRenderAsset^ targetRendererMemento1;
		TargetRenderAsset^ targetRendererMemento2;
		TargetRenderAsset^ targetRendererMemento3;
		CudaRaytraceRenderAsset^ cudaRenderer;
		CudaVideoRenderAsset^ cudaVideoRenderer;
		CudaFileRenderAsset^ cudaFileRenderer;
		SceneGraphFactory^ factory;
		TextureLayerNode^ layer;
		EffectAsset^ effectAssetTarget;
		EffectAsset^ effectAssetMemento;
		EffectAsset^ effectAssetMemento1;
		EffectAsset^ effectAssetMemento2;
		EffectAsset^ effectAssetViewLayer;
		EffectAsset^ effectAssetDeferred;
		double deferredTime;
	private: TreeNode^ BaseNode_getTreeNode(BaseNode^ a_pBaseNode, System::Windows::Forms::TreeNode^ treeNode)
			 {
				 BaseNode^ nextNode=a_pBaseNode;
				 while(nextNode){
					 if(!nextNode->GetChild())
						 treeNode->Nodes->Add(nextNode->ToString()+" "+nextNode->GetHashCode());
					 else{
						 System::Windows::Forms::TreeNode^ childTreeNode=
							 gcnew System::Windows::Forms::TreeNode(nextNode->ToString()+" "+nextNode->GetHashCode());
						 treeNode->Nodes->Add(BaseNode_getTreeNode(nextNode->GetChild(), childTreeNode));
					 }
					 nextNode=nextNode->GetNext();
				 }
				 return treeNode;
			 }
	private: System::Void trackBar1_Scroll(System::Object^  sender, System::EventArgs^  e)
			 {
				 double trackBarValue=this->trackBar1->Value;
				 double s=trackBarValue/1000.0;

				 ////transformNode2->SetMatrixWorld(
				 //// Math::Cos(s)*1.0/s,-Math::Sin(s)*1.0/s,0,0,
				 //// Math::Sin(s)*1.0/s, Math::Cos(s)*1.0/s,0,0,
				 //// 0,0,.5,0,
				 //// 0,0,1,1);

				 //transformNode->SetMatrixWorld(
				 // 1,0,0,0,
				 // 0,1,0,0,
				 // 0,0,1,0,
				 // s*0.5,0,0,1);

				 double z=1.0;
				 s=s*7;
				 transformNode->SetMatrixWorld(
				  z,0,0,0,
				  0, Math::Cos(s)*z,-Math::Sin(s)*z,0,
				  0, Math::Sin(s)*z, Math::Cos(s)*z,0,
				  0,0,0,1);

				 transformNode2->SetMatrixWorld(
				  Math::Cos(s)*z,0, -Math::Sin(s)*z,0,
				  0, z,0,0,
				  Math::Sin(s)*z, 0, Math::Cos(s)*z,0,
				  0, 0,0,1);

				 //System::DateTime startTime=System::DateTime::Now;

				 //effectAssetTarget->SetValue("Time", s);
				 Panel1_OnPaint(this, nullptr);
				 //this->toolStripStatusLabel1->Text=""+trackBarValue;				 
			 }
	private: System::Void treeView1_AfterSelect(System::Object^  sender, System::Windows::Forms::TreeViewEventArgs^  e) {
				 this->statusStrip1->Text=e->ToString();
			 }
	private: System::Void treeView1_DragDrop(System::Object^  sender, System::Windows::Forms::DragEventArgs^  e) {
				 this->statusStrip1->Text=e->ToString();
			 }
	private: System::Void treeView1_ItemDrag(System::Object^  sender, System::Windows::Forms::ItemDragEventArgs^  e) {
				 this->statusStrip1->Text=e->ToString();
			 }
	private: System::Void openToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
				 openFileDialog1->Filter = "all known|*.sdkmesh;*.obj|sdkmesh files (*.sdkmesh)|*.sdkmesh|OBJ files (*.obj)|*.obj|All files (*.*)|*.*" ;
				 this->openFileDialog1->ShowDialog();

				 if(this->openFileDialog1->SafeFileName->ToLower()->EndsWith(".sdkmesh"))
				 {
					 drawMockNode = factory->CreateDrawMock();
					 openFileDialog2->Filter = "FX|*.fx|All files (*.*)|*.*" ;
					 effectAssetTarget = factory->CreateEffect("Default.DirectX9.sdkmesh.fx");
					 SdkMeshAsset^ sdkMeshAsset = factory->CreateSdkMesh(this->openFileDialog1->SafeFileName);
					 drawMockNode->SetMesh(sdkMeshAsset);
					 drawMockNode->MakeParent(this->transformNode2);
				 }
				 else if(this->openFileDialog1->SafeFileName->ToLower()->EndsWith(".obj"))
				 {
					 drawMockNode = factory->CreateDrawMock();
					 //openFileDialog2->Filter = "FX|*.fx|All files (*.*)|*.*" ;
					 //openFileDialog2->ShowDialog();
					 //effectAssetTarget = factory->CreateEffect(this->openFileDialog2->SafeFileName);					 				
					 //effectAssetTarget = factory->CreateEffect("Position.DirectX9.obj.fx");					 				
					 effectAssetTarget->SetValue("Time", 1.0);
					 effectAssetMemento->SetValue("Time", 1.0);
					 effectAssetMemento1->SetValue("Time", 1.0);
					 effectAssetMemento2->SetValue("Time", 1.0);
					 ObjMeshAsset^ objMeshAsset = factory->CreateObjMesh(this->openFileDialog1->FileName);
					 drawMockNode->SetMesh(objMeshAsset);
					 drawMockNode->MakeParent(this->transformNode2);
				 } else if(	this->openFileDialog1->SafeFileName->ToLower()->EndsWith(".m4v")
					 ||		this->openFileDialog1->SafeFileName->ToLower()->EndsWith(".m2v")) {
					 this->cudaVideoRenderer = factory->CreateCudaVideoRenderAsset(this->openFileDialog1->FileName);

					 TextureLayerNode^ textureLayerNode = factory->CreateTextureLayer(cudaVideoRenderer);
					 textureLayerNode->MakeParent(layerRoot);
				 }

				 this->treeView1->Nodes->Clear();
				 this->treeView1->Nodes->Add(
					 this->BaseNode_getTreeNode(targetRoot, gcnew TreeNode("Scene")));

				 Panel1_OnPaint(this, nullptr);
			 }
	private: System::Void Panel1_OnPaint(System::Object^  sender, System::Windows::Forms::PaintEventArgs^  e) {				 			 				 
				 if(this->checkBox1->Checked)
					this->targetRoot->Memento();
				 this->targetRoot->ApplyCamera(cameraAsset);
				 //if(this->checkBox1->Checked)
				 if(this->targetRendererMemento0!=nullptr)
						 this->targetRendererMemento0->Present(effectAssetMemento);
				 if(this->checkBox1->Checked)
				 {
					 if(this->targetRenderer!=nullptr)
					    this->targetRenderer->Present(effectAssetTarget);					 
					 if(this->cudaRenderer!=nullptr)
						this->cudaRenderer->Present(50);		
					 //if(this->targetRendererMemento1!=nullptr)
						// this->targetRendererMemento1->Present(effectAssetMemento1);
					 //if(this->targetRendererMemento2!=nullptr)
						// this->targetRendererMemento2->Present(effectAssetMemento2);
				 }
				 if(this->cudaRenderer!=nullptr)
					 this->cudaRenderer->Present((int)this->numericUpDown1->Value);				 
				 if(this->cudaVideoRenderer!=nullptr)
					this->cudaVideoRenderer->Present(1);
				 if(this->viewLayerRenderer!=nullptr)
					 this->viewLayerRenderer->Present(effectAssetViewLayer);
			 }
	private: System::Void Panel1_OnMouseClick(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) 
			 {
				 switch ( e->Button )
				 {
				 case System::Windows::Forms::MouseButtons::Middle:
					 {
						 Panel1_OnPaint(this, nullptr);
					 }
					 break;
				 case System::Windows::Forms::MouseButtons::Right:
					 {
						 PickIntersection^ pick=this->targetRoot->TestIntersection(this->panel1->Handle);
						 this->toolStripStatusLabel1->Text=
							 "face: "+pick->face
							 +" texU: "+pick->texture_u
							 +" texV: "+pick->texture_v
							 +" dist: "+pick->distance
							 +" bary1: "+pick->barycentric1
							 +" bary2: "+pick->barycentric2
							 +" backface: "+pick->backface;
					 }
					 break;
				 case System::Windows::Forms::MouseButtons::Left:
					 {
						 double eyePtX=e->X-(RESOLUTION_X/2.0)+.5;
						 double eyePtY=e->Y-(RESOLUTION_Y/2.0)+.5;
						 double eyePtZ=.18;//*Math::Sin(DateTime::Now.Millisecond*(1.0/1000.0)*Math::PI*2);						
						 
						 this->cameraAsset->Perspective(74.0, (double)RESOLUTION_X/(double)RESOLUTION_Y, 0.01, 0.0);
						 this->cameraAsset->LookAt(eyePtX*.0051732894,eyePtZ,eyePtY*.0051732894,	
							 0.00123789, 0.223748590456,0.001435789,		0.01234,0.9934256,0.00964578);

						 //this->cameraAsset->Orthogonal(128.0, 72.0, 0.1, 1000.0);
						 //this->cameraAsset->LookAt(64,100,-36,	64,-10,-36,		0,0,-1);
						 //this->cameraAsset->LookAt(62.4,100,-34.4,	62.4,-10,-34.4,		0,0,-1);

						 this->toolStripStatusLabel1->Text="x: "+e->X+" y:"+e->Y;
						 Panel1_OnPaint(this, nullptr);
					 }
					 break;
				 default:
					 break;
				 }
			 }
	private: System::Void Panel1_OnMouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e) 
			 {
				 switch ( e->Button )
				 {
				 case System::Windows::Forms::MouseButtons::Middle:
					 {
						 Panel1_OnPaint(this, nullptr);
					 }
					 break;
				 case System::Windows::Forms::MouseButtons::Right:
					 {
						 PickIntersection^ pick=this->targetRoot->TestIntersection(this->panel1->Handle);
						 this->toolStripStatusLabel1->Text=
							 "face: "+pick->face
							 +" texU: "+pick->texture_u
							 +" texV: "+pick->texture_v
							 +" dist: "+pick->distance
							 +" bary1: "+pick->barycentric1
							 +" bary2: "+pick->barycentric2
							 +" backface: "+pick->backface;
					 }
					 break;
				 case System::Windows::Forms::MouseButtons::Left:
					 {
						 double eyePtX=e->X-(RESOLUTION_X/2.0)+.5;
						 double eyePtY=e->Y-(RESOLUTION_Y/2.0)+.5;
						 double eyePtZ=0.18;//*Math::Sin(DateTime::Now.Millisecond*(1.0/1000.0)*Math::PI*2);						
						 
						 this->cameraAsset->Perspective(74.0, (double)RESOLUTION_X/(double)RESOLUTION_Y, 0.01, 0.0);
						 this->cameraAsset->LookAt(eyePtX*.0051732894,eyePtZ,eyePtY*.0051732894,	
							 0.00123789, 0.223748590456,0.001435789,		0.01234,0.9934256,0.00964578);

						 //this->cameraAsset->Orthogonal(128.0, 72.0, 0.1, 1000.0);
						 //this->cameraAsset->LookAt(64,100,-36,	64,-10,-36,		0,0,-1);
						 //this->cameraAsset->LookAt(62.4,100,-34.4,	62.4,-10,-34.4,		0,0,-1);

						 this->toolStripStatusLabel1->Text="x: "+e->X+" y:"+e->Y;						 
						 Panel1_OnPaint(this, nullptr);
					 }
					 break;
				 default:
					 break;
				 }
			 }
	private: System::Void renderTargetToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
				 TextureLayerNode^ textureLayerNode = factory->CreateTextureLayer(targetRenderer);			 				 
				 textureLayerNode->MakeParent(layerRoot);

				 if(this->cudaRenderer!=nullptr) {
					 this->cudaRenderer->SetTextureTarget(4, factory->CreateTextureAsset(targetRenderer));
					 this->cudaRenderer->SetTextureTarget(5, factory->CreateTextureAsset(targetRendererMemento0));
					 this->cudaRenderer->SetTextureTarget(6, factory->CreateTextureAsset(targetRendererMemento1));
					 this->cudaRenderer->SetTextureTarget(7, factory->CreateTextureAsset(targetRendererMemento2));
				 }

				 Panel1_OnPaint(this, nullptr);

				 this->treeView3->Nodes->Clear();
				 this->treeView3->Nodes->Add(
					 this->BaseNode_getTreeNode(layerRoot, gcnew TreeNode("Layer")));
			 }
	private: System::Void raytraceCPUToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
				 System::DateTime startTime=System::DateTime::Now;
				 RaytraceRenderAsset^ raytracer=factory->CreateRaytracer();
				 raytracer->SetRoot(drawMockNode);
				 //TODO: insert effect code for raytracer
				 raytracer->Present(nullptr);
				 layer=factory->CreateTextureLayer(raytracer);

				 layer->MakeParent( targetRoot );

				 Panel1_OnPaint(this, nullptr);

				 this->toolStripStatusLabel1->Text=(System::DateTime::Now-startTime).ToString();

				 this->treeView3->Nodes->Clear();
				 this->treeView3->Nodes->Add(
					 this->BaseNode_getTreeNode(layerRoot, gcnew TreeNode("Layer")));
			 }
	private: System::Void raytraceGPUToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
				 TextureLayerNode^ textureLayerNode = factory->CreateTextureLayer(cudaRenderer);

				 cudaRenderer->SetRoot(drawMockNode);

				 textureLayerNode->MakeParent(layerRoot);

				 this->treeView3->Nodes->Clear();
				 this->treeView3->Nodes->Add(
					 this->BaseNode_getTreeNode(layerRoot, gcnew TreeNode("Layer")));

				 Panel1_OnPaint(this, nullptr);
			 }
	private: System::Void xGPUToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
				 System::DateTime startTime=System::DateTime::Now;
				 for(int i=0; i<100; i++)
					 this->cudaRenderer->Present((int)this->numericUpDown1->Value);

				 Panel1_OnPaint(this, nullptr);
				 this->toolStripStatusLabel1->Text="CUDA "+(System::DateTime::Now-startTime).ToString();
			 }
	private: System::Void xDXToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
				 System::DateTime startTime=System::DateTime::Now;
				 for(int i=0; i<100; i++)
					 this->targetRenderer->Present(effectAssetTarget);

				 Panel1_OnPaint(this, nullptr);
				 this->toolStripStatusLabel1->Text="DirectX "+(System::DateTime::Now-startTime).ToString();
			 }
	private: System::Void button1_Click(System::Object^  sender, System::EventArgs^  e) { 
				 for(int i=0; i<64; i++)
				 {
					 this->targetRenderer->Present(effectAssetTarget);
					 this->cudaRenderer->Present((int)this->numericUpDown1->Value);
					 this->viewLayerRenderer->Present(effectAssetViewLayer);
				 }
			 }

			 //delegate System::Void backgroundWorker1_DoWork(System::Object^  sender, System::ComponentModel::DoWorkEventArgs^  e);
	private: delegate System::Void SetAOCallback(System::Void);
	private: System::Void SetAO(System::Void) {
				 this->cudaRenderer->Present(1);
				 //this->viewLayerRenderer->Present();
			 }
	private: System::Void backgroundWorker1_RunCompleted(System::Object^  sender, System::ComponentModel::RunWorkerCompletedEventArgs^ e){
				 this->backgroundWorker1->RunWorkerAsync();
			 }
	private: System::Void backgroundWorker1_DoWork(System::Object^  sender, System::ComponentModel::DoWorkEventArgs^  e) {
				 if(this->InvokeRequired)
				 {
					 SetAOCallback^ d = gcnew SetAOCallback(this, &Form1::SetAO);
					 this->Invoke(d, gcnew array<System::Object^> {});
				 }
				 else
					 SetAO();
			 }
	private: System::Void deferedLightingToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
				 TextureLayerNode^ textureLayerNode = factory->CreateTextureLayer(this->cudaRenderer);
				 effectAssetDeferred = factory->CreateEffect("TextureLayer.Deferred.Lighting.DirectX9.fx");
				 //TODO: find way to handle effects, a map?
				 effectAssetViewLayer=effectAssetDeferred;
				 textureLayerNode->MakeParent(layerRoot);

				 this->treeView3->Nodes->Clear();
				 this->treeView3->Nodes->Add(
					 this->BaseNode_getTreeNode(layerRoot, gcnew TreeNode("Layer")));
				 timer1->Interval=10;
				 timer1->Start();				 				 
				 Panel1_OnPaint(this, nullptr);
				 //layer=textureLayerNode;
			 }
	private: System::Void timer1_Tick(System::Object^  sender, System::EventArgs^  e) {
				 deferredTime+=0.003;
				 if(effectAssetDeferred!=nullptr)
					 effectAssetDeferred->SetValue("Time", deferredTime);	
				 Panel1_OnPaint(this, nullptr);
			 }
	private: System::Void combineToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
				 TextureLayerNode^ textureLayerNode = factory->CreateTextureLayer(this->cudaRenderer);
				 effectAssetDeferred = factory->CreateEffect("TextureLayer.Combine.Multiply.DirectX9.fx");

				 textureLayerNode->MakeParent(layerRoot);

				 this->treeView3->Nodes->Clear();
				 this->treeView3->Nodes->Add(
					 this->BaseNode_getTreeNode(layerRoot, gcnew TreeNode("Layer")));
				 timer1->Interval=10;
				 timer1->Start();				 				 
				 Panel1_OnPaint(this, nullptr);
			 }
	private: System::Void checkBox1_CheckedChanged(System::Object^  sender, System::EventArgs^  e) {
				 //cameraAsset->LookAt(0,0,2,0,0,0,0,1,0);
				 //this->targetRoot->ApplyCamera(cameraAsset);
				 //this->targetRendererMemento0->Present(effectAssetMemento);
				 //this->targetRoot->Memento();
				 //this->targetRenderer->Present(effectAssetTarget);
				 //this->targetRendererMemento1->Present(effectAssetMemento1);
				 //this->targetRendererMemento2->Present(effectAssetMemento2);					 
				 //this->cudaRenderer->Present((int)this->numericUpDown1->Value);				 
				 //this->viewLayerRenderer->Present(effectAssetViewLayer);
			 }
	private: System::Void videoToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
				 TextureLayerNode^ textureLayerNode = factory->CreateTextureLayer(cudaVideoRenderer);

				 textureLayerNode->MakeParent(layerRoot);

				 this->treeView3->Nodes->Clear();
				 this->treeView3->Nodes->Add(
					 this->BaseNode_getTreeNode(layerRoot, gcnew TreeNode("Layer")));

				 Panel1_OnPaint(this, nullptr);
			 }
	private: System::Void saveScreenshotToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
				 saveFileDialog1->ShowDialog();
				 Bitmap^ bitmap = factory->CreateBitmap(factory->CreateTextureAsset(targetRenderer));
				 //Imaging::ImageCodecInfo::
				 bitmap->Save(saveFileDialog1->FileName );
			 }
};
}

