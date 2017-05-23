using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.NVRTC;

public class SPH : CudaGenerator
{
    // 粒子の状態
    struct pSPH
    {
        public Vector3 pos;
        public Vector3 vel;
        public float m;
        public float rho;
        public float _;
        public float col;
        public pSPH(Vector3 _pos, Vector3 _vel, float __)
        {
            pos = _pos;
            vel = _vel;
            m = 0.15f;
            rho = 0;
            _ = __;
            col = 0.5f;
        }
    }

    // 粒子数
    public int NUM_OF_P;
    int NUM_OF_MC = 64 * 64 * 64;

    public float h;

    // レンダリングシェーダー
    public Shader SPHShader;
    public Shader MCShader;
    // マテリアル
    Material SPHMaterial;
    Material MCMaterial;
    // コンピュートバッファ
    ComputeBuffer computeBuffer;
    ComputeBuffer MCBuf;

    CudaDeviceVariable<pSPH> d_pSPHArr;
    CudaDeviceVariable<Vector4> d_MCBuf;

    //重力加速度
    public float g;
    //水の一辺の長さ
    public int water_len;
    // delta time
    float dt;

    public bool MC;
    public float MC_S;
    public float MC_L;

    pSPH[] h_pSPHArr;
    Vector4[] h_MCBuf;


    // Use this for initialization
    void Start()
    {
        InitializeCUDA();

        SPHMaterial = new Material(SPHShader);
        MCMaterial = new Material(MCShader);

        h_pSPHArr = new pSPH[NUM_OF_P];
        h_MCBuf = new Vector4[NUM_OF_MC];
        dt = Time.deltaTime;

        int p = 0;
        int p_TH = 8000;
        for (int i = 0; i < NUM_OF_P; i++)
        {
            //wall
            if (i < p_TH)
            {
                int x, y, z;
                do
                {
                    x = p % 64;
                    z = (p / 64) % 64;
                    y = (p / 64 / 64) % 100;
                    p++;
                } while (!(y == 0 | x == 0 | z == 0 | x == 63 | z == 63) | (1 < 0 & 10 < x & x < 20 & 10 < z & z < 20));
                h_pSPHArr[i] = new pSPH(
                    new Vector3(x, y, z) * (0.3f) + new Vector3(1f, 1f, 1f),
                    new Vector3(0, 0, 0),
                    0f //* (float)(x+z) / (128f) :0.4:
                );
            }
            else if (p_TH <= i)
            {
                int j = i - p_TH;
                int x, y, z;
                {
                    p = j;

                    x = p % water_len;
                    z = (p / water_len) % water_len;
                    y = (p / water_len / water_len);

                    /*
                    x = p % 16;
                    z = (p / 16) % 16;
                    y = (p / 16 / 16) % 1000;
                    */
                    /*
                    x = p % 24;
                    z = (p / 24) % 24;
                    y = (p / 24 / 24) % 1000;
                    */
                }
                h_pSPHArr[i] = new pSPH(
                    /*new Vector3(Random.Range(-10.0f, 10.0f), Random.Range(-10.0f, 10.0f), Random.Range(-10.0f, 10.0f)),*/
                    new Vector3(x, y, z) * 0.4f + new Vector3(0, 4.0f, 0) + new Vector3(5f, 0f, 5f),
                    new Vector3(0, 0, 0),
                    1.0f
                );
            }
        }
        for (int i = 0; i < NUM_OF_MC; ++i)
        {
            h_MCBuf[i] = new Vector4(0f, 0f, 0f, 0f);
        }

        d_pSPHArr = h_pSPHArr;
        d_MCBuf = h_MCBuf;

        computeBuffer = new ComputeBuffer(NUM_OF_P, Marshal.SizeOf(typeof(pSPH)));
        MCBuf = new ComputeBuffer(NUM_OF_MC, Marshal.SizeOf(typeof(Vector4)));

        int threadsPerBlock = 1024;
        int blocksPerGrid = (NUM_OF_P + threadsPerBlock - 1) / threadsPerBlock;
        cudaKernel[0].BlockDimensions = new dim3(threadsPerBlock, 1, 1);
        cudaKernel[0].GridDimensions = new dim3(blocksPerGrid, 1, 1);
        cudaKernel[1].BlockDimensions = new dim3(threadsPerBlock, 1, 1);
        cudaKernel[1].GridDimensions = new dim3(blocksPerGrid, 1, 1); 
        cudaKernel[2].BlockDimensions = new dim3(1024, 1, 1);
        cudaKernel[2].GridDimensions = new dim3(NUM_OF_MC / 1024, 1, 1);
    }

    void OnDisable()
    {
        d_pSPHArr.Dispose();
        d_MCBuf.Dispose();

        computeBuffer.Release();
        MCBuf.Release();
    }

    // Update is called once per frame
    void Update()
    {
        cudaKernel[0].Run(d_pSPHArr.DevicePointer, h, NUM_OF_P);
        cudaKernel[1].Run(d_pSPHArr.DevicePointer, h, g, dt, NUM_OF_P);
        if (MC)
        {
            cudaKernel[2].Run(d_pSPHArr.DevicePointer, d_MCBuf.DevicePointer, (int)MC_S, MC_L, h, NUM_OF_P, NUM_OF_MC);
            d_MCBuf.CopyToHost(h_MCBuf);
            MCBuf.SetData(h_MCBuf);
        }
        else
        {
            d_pSPHArr.CopyToHost(h_pSPHArr);
            computeBuffer.SetData(h_pSPHArr);
        }
    }

    void OnRenderObject()
    {
        // テクスチャ、バッファをマテリアルに設定
        // レンダリングを開始
        if (MC)
        {
            MCMaterial.SetBuffer("MCBuf", MCBuf);
            MCMaterial.SetFloat("M", NUM_OF_MC);
            MCMaterial.SetFloat("_GS", MC_S);
            MCMaterial.SetFloat("_GL", MC_L);
            MCMaterial.SetPass(0);
            Graphics.DrawProcedural(MeshTopology.Points, NUM_OF_MC);
        }
        else
        {
            SPHMaterial.SetBuffer("pSPHBuf", computeBuffer);
            SPHMaterial.SetPass(0);
            Graphics.DrawProcedural(MeshTopology.Points, NUM_OF_P);
        }
    }
}

