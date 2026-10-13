// Original C3 Farquhar / Ball--Berry implementation. See CLM_PHOTOSYNTHESIS.md.
// One independent leaf per thread; AoS input (n,7), output (n,12).
// NVRTC provides device math builtins without a host math.h include path.

__device__ float arrhenius(float t, float activation) {
    return expf((activation / (float)(8.314462618 * 298.15)) * (1.f - 298.15f / t));
}

__device__ float peaked(float t, float activation, float deactivation, float entropy) {
    float numerator = 1.f + expf((298.15f * entropy - deactivation) / (float)(8.314462618 * 298.15));
    float denominator = 1.f + expf((t * entropy - deactivation) / (8.314462618f * t));
    return arrhenius(t, activation) * numerator / denominator;
}

struct LeafEnvironment {
    float ca, rh, gb, stress, vcmax, product, rd, kc, ko, gamma, transport;
};

__device__ void evaluate(float ci, const LeafEnvironment& e, const float* p, float* v) {
    float carbon = fmaxf(ci-e.gamma, 0.f);
    float ac = e.vcmax*carbon/(ci+e.kc*(1.f+p[9]/e.ko));
    float aj = e.transport*carbon/(4.f*ci+8.f*e.gamma);
    float gross = fminf(fminf(ac, aj), e.product);
    float net = gross-e.rd;
    float cs = fmaxf(e.ca-1.4f*net/e.gb, 1e-9f);
    float x = p[6]*fmaxf(net, 0.f)*e.stress/cs;
    float b = e.gb-p[7]-x;
    float c = e.gb*(p[7]+x*fminf(e.rh, 1.f));
    float root = sqrtf(b*b+4.f*c);
    float gs = b >= 0.f ? 2.f*c/fmaxf(b+root, 1e-30f) : .5f*(root-b);
    v[0]=gross; v[1]=net; v[2]=ac; v[3]=aj; v[4]=cs; v[5]=gs;
    v[6]=net-(e.ca-ci)/(1.4f/e.gb+1.6f/gs);
}

extern "C" __global__ void run_leaves(const float* u, const float* p, float* output, int n) {
    for (int row=blockIdx.x*blockDim.x+threadIdx.x; row<n; row+=blockDim.x*gridDim.x) {
        const float* v=u+7*row;
        float par=v[0], t=v[1], nitrogen=v[5];
        float growth=fminf(fmaxf(p[8]-273.15f, 11.f), 35.f);
        LeafEnvironment e;
        e.ca=v[2]; e.rh=v[3]; e.gb=v[4]; e.stress=v[6];
        float vc_factor=peaked(t, 72000.f, 200000.f, 668.39f-1.07f*growth);
        e.vcmax=p[0]*nitrogen*vc_factor;
        float jmax=p[1]*nitrogen*peaked(t, 50000.f, 200000.f, 659.70f-.75f*growth);
        e.product=3.f*p[2]*nitrogen*vc_factor;
        e.rd=p[3]*peaked(t, 46390.f, 150650.f, 490.f);
        e.kc=404.9f*arrhenius(t, 79430.f);
        e.ko=278400.f*arrhenius(t, 36380.f);
        e.gamma=42.75f*arrhenius(t, 37830.f);
        float light=.5f*p[4]*par;
        float total=light+jmax;
        float den=total+sqrtf(fmaxf(total*total-4.f*p[5]*light*jmax, 0.f));
        e.transport=den>0.f ? 2.f*light*jmax/fmaxf(den, 1e-30f) : 0.f;
        float lo=0.f, hi=e.ca+e.rd*(1.4f/e.gb+1.6f/p[7])+1.f;
        float rates[7];
        for (int iter=0; iter<48; ++iter) {
            float mid=.5f*(lo+hi);
            evaluate(mid, e, p, rates);
            if (rates[6]<0.f) lo=mid;
            else hi=mid;
        }
        float ci=.5f*(lo+hi);
        evaluate(ci, e, p, rates);
        float* out=output+12*row;
        out[0]=rates[0]; out[1]=rates[1]; out[2]=rates[2]; out[3]=rates[3];
        out[4]=e.product; out[5]=e.rd; out[6]=e.transport; out[7]=ci;
        out[8]=rates[4]; out[9]=rates[5];
        out[10]=(e.gb*fminf(e.rh, 1.f)+rates[5])/(e.gb+rates[5]);
        out[11]=rates[6];
    }
}
