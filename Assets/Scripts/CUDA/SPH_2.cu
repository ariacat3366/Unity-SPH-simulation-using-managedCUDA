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
__device__ float3 Gradient_Kernel_Spiky(float3 r, float h) {
	float PI = 3.14159;
    float _r = sqrt(length(r));
    float v = -45.0f / (PI * pow(h, 6) * _r) * pow(h - _r, 2);
	return make_float3(r.x*v, r.y*v, r.z*v);
}
__device__ float Lap_Kernel_Viscosity(float3 r, float h) {
	float PI = 3.14159;
	return 45.0f / (PI * pow(h, 5)) * (1 - sqrt(length(r)) / h);
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
SPH_2(pSPH *p, const float h, const float g, const float t, const int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx > N) return;

    pSPH _p = p[idx];
    if (_p._ <= 0.2) return;

    float rho_0 = 1.0f;
    float y = 5;
    float p_i = rho_0 * pow(_p.rho/rho_0 - 1.0f, y);
    float p_j;
    float3 F_p = make_float3(0,0,0);
    float3 F_v = make_float3(0,0,0);
    float3 F_ex = make_float3(0,0,0);
    float3 G_cs = make_float3(0,0,0);
    float L_cs = 0.0f;
    float3 accel = make_float3(0,0,0);
    float3 gravity = make_float3(0, g, 0);

    int i;
    for (i = 0; i < N; ++i)
    {
        pSPH __p = p[i];
        float3 r = dif_float3(_p.pos, __p.pos);
        if (i == idx) continue;
        if (length(r) > h*h) continue;

        float scale_p = 1.0f;
        float scale_v = 1.0f;
        //wall
        if (__p._ <= 0.2)
        {
             scale_p = 2.0f;
             scale_v = 2.0f;
        }

        p_j = rho_0 * pow(__p.rho/rho_0 - 1.0f,y);
        F_p = add_float3(F_p, scale_float3(scale_p * -1.0f * __p.m * (p_i + p_j) / (2.0f*__p.rho), Gradient_Kernel_Spiky(r, h)));
        F_v = add_float3(F_v, scale_float3(scale_v * Lap_Kernel_Viscosity(r, h), scale_float3(0.1f * __p.m, dif_float3(__p.vel, _p.vel))));
       
        //G_cs = add_float3(G_cs, scale_float3(__p.m, Gradient_Kernel_Poly6(r, h)));
        //L_cs = __p.m * Lap_Kernel_Poly6(r, h);

        
    }
    
    if (L_cs > 0.01) 
    {
        //F_ex = dif_float3(F_ex, scale_float3(0.002f * L_cs / length(G_cs), G_cs));
    }

    accel = add_float3(accel, gravity);
    accel = add_float3(accel, F_ex);
    accel = add_float3(accel, scale_float3(rho_0, add_float3(F_p, F_v)));
    //p[idx].vel = add_float3(_p.vel, scale_float3(0.01f, accel));
    //p[idx].pos = add_float3(_p.pos, scale_float3(0.01f, p[idx].vel));
    p[idx].vel = add_float3(_p.vel, scale_float3(t, accel));
    p[idx].pos = add_float3(_p.pos, scale_float3(t, p[idx].vel));

    return;
}
