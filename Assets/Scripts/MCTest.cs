using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MCTest : MonoBehaviour {


    public Shader shader;
    Material material;

    const int PG = 256;
    int ParticleCount = PG * PG * PG;

    // Use this for initialization
    void Start()
    {
        material = new Material(shader);
    }

    // Update is called once per frame
    void Update()
    {

    }

    void OnRenderObject()
    {
        //material.SetBuffer("pSPHBuf", Buffer_pSPHIn);
        material.SetPass(0);
        Graphics.DrawProcedural(MeshTopology.Points, ParticleCount);
    }
}
