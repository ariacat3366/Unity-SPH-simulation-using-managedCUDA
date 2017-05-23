//function kernel
__device__ float length(float3 r) {
    return r.x*r.x + r.y*r.y + r.z*r.z;
}
__device__ float3 mul_float3(float3 r1, float3 r2) {
    return make_float3(r1.x * r2.x,  r1.y * r2.y,  r1.z * r2.z);
}
__device__ float3 add_float3(float3 r1, float3 r2) {
    return make_float3(r1.x + r2.x,  r1.y + r2.y,  r1.z + r2.z);
}
__device__ float3 dif_float3(float3 r1, float3 r2) {
    return make_float3(r1.x - r2.x,  r1.y - r2.y,  r1.z - r2.z);
}
__device__ float3 scale_float3(float s, float3 r) {
    r.x *= s;
    r.y *= s;
    r.z *= s;
    return r;
}
__device__ float Kernel_Poly6(float3 r, float h) {
    float PI = 3.14159;
    return 315.0f / (64 * PI * pow(h, 9)) * pow(pow(h, 2) - length(r), 3);
}
__device__ float3 Gradient_Kernel_Poly6(float3 r, float h) {
    float PI = 3.14159;
    return make_float3(
            r.x * -945.0f / ( 32.0f * PI * pow(h,9) ) * pow(pow(h, 2) - length(r), 2),
            r.y * -945.0f / ( 32.0f * PI * pow(h,9) ) * pow(pow(h, 2) - length(r), 2),
            r.z * -945.0f / ( 32.0f * PI * pow(h,9) ) * pow(pow(h, 2) - length(r), 2));
}
__device__ float Lap_Kernel_Poly6(float3 r, float h) {
    float PI = 3.14159;
    return 945.0f / (8 * PI * pow(h, 9)) * (pow(h, 2) - length(r)) * (length(r) - 3 / 4 * (pow(h, 2) - length(r)));
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
MC(pSPH *p, float4 *MCBuf, const int len, const float scale, const float h, const int N, const int M)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx > M) return;

    int px = idx%len;
    int py = (idx/len)%len;
    int pz = (idx/len/len);
    float3 pos = make_float3(px*scale,py*scale,pz*scale);

    float rho = 0;
    float3 grad_rho = make_float3(0,0,0);

    int i;
    for (i = 0; i < N; ++i)
    {
        pSPH _p = p[i];
        float3 r = dif_float3(pos, _p.pos);
        if (i == idx) continue;
        if (length(r) > h*h) continue;
        if (_p._ <= 0.5f) continue;
        rho += _p.m * Kernel_Poly6(r, h);
        grad_rho = add_float3(grad_rho, scale_float3(_p.m * -1.0f, Gradient_Kernel_Poly6(r, h)));
    }
    float4 _MCBuf = make_float4(grad_rho.x, grad_rho.y, grad_rho.z, rho);
    MCBuf[idx] = _MCBuf;
    return;
}
