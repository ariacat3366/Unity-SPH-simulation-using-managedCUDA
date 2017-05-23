//function kernel
__device__ float length(float3 r) {
    return r.x*r.x + r.y*r.y + r.z*r.z;
}
__device__ float3 dif_float3(float3 r1, float3 r2) {
    return make_float3(r1.x-r2.x, r1.y-r2.y, r1.z-r2.z);
}
__device__ float Kernel_Poly6(float3 r, float h) {
	float PI = 3.14159;
	return 315.0f / (64 * PI * pow(h, 9)) * pow(pow(h, 2) - length(r), 3);
}


//SPH particle struct
struct pSPH {
	float3 pos;
	float3 vel;
    float m;
	float rho;
	float _;
	float col;
};


extern "C" __global__ void
SPH_1(pSPH *p, const float h, const int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx > N) return;
    p[idx].rho = 0.0f;
    pSPH _p = p[idx];
    int i;
    float _rho;
    for (i = 0; i < N; ++i)
    {
        if (i == idx) continue;
        float3 r = dif_float3(_p.pos, p[i].pos);
        if (length(r) <= h*h)
        {
            _rho += p[i].m * Kernel_Poly6(r, h);
            
        }
    }
    p[idx].rho = _rho + 0.0001f;

    if (_p._ <= 0.2) p[idx].col = 1.0f / (p[idx].rho + 1.0f) - 0.1f;
    else p[idx].col = 1.0f / (p[idx].rho/1.4f + 1.0f);

    return;
}
