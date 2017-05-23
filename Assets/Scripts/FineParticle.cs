using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NVRTC;

public class FineParticle : CudaGenerator
{
    // 粒子の状態
    struct Particle
    {
        public Vector3 pos;
        public Vector3 vel;
        public float m;
        public float rho;
        public float _;
        public float col;
    }

    // 粒子数
    public int NUM_OF_P;
    public float  h;

    // レンダリングシェーダー
    public Shader FineParticleShader;
    // マテリアル
    Material FineParticleMaterial;
    // コンピュートバッファ
    ComputeBuffer computeBuffer;

    CudaDeviceVariable<Particle> d_P;

  

    // 粒子の数は，1cm^3に約8.0e6個 (800万個)
    //50 micro m
    // 質量
    float m = 5e-7f; 
    // バネ定数
    float k = 1.0f;
    // 　劉氏の反発係数
    float e = 1.0f;
    // 重力加速度
    public float g = -9.8f;
    // 粘着力
    float tau_c = 1.0f;
    // 引っ張り強度
    float d_t = 1.0f;
    // delta time
    float dt;

    Particle[] h_P;

    // Use this for initialization
    void Start()
    {
        InitializeCUDA();

        FineParticleMaterial = new Material(FineParticleShader);

        h_P = new Particle[NUM_OF_P];

        d_P = h_P;

        computeBuffer = new ComputeBuffer(NUM_OF_P, Marshal.SizeOf(typeof(Particle)));

        //8.0e12 ( = 1m^3) =~ 2^43
        cudaKernel[0].BlockDimensions = new dim3(2 ^ 10, 1, 1); //max 2048:(2^10, 2, 1)
        cudaKernel[0].GridDimensions = new dim3(2 ^ 20, 1, 1); //max (2^31-1, 2^16-1, 2^16-1) 1.8e22個
        
       
    }

    void OnDisable()
    {
        d_P.Dispose();
        
        computeBuffer.Release();
    }

    // Update is called once per frame
    void Update()
    {
        cudaKernel[0].Run(d_P.DevicePointer, h, NUM_OF_P);
        d_P.CopyToHost(h_P);
        computeBuffer.SetData(h_P);
        
    }

    void OnRenderObject()
    {
        // テクスチャ、バッファをマテリアルに設定
        FineParticleMaterial.SetBuffer("p", computeBuffer);
        FineParticleMaterial.SetPass(0);
        
        // レンダリングを開始
        Graphics.DrawProcedural(MeshTopology.Points, NUM_OF_P);
    }
}

