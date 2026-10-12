// [未实测] 本机无 nvcc；源码按 CUDA C++ Programming Guide 的 grid-stride
// loop 与合并访问规则组织，需在带 CUDA Toolkit 的 xuanze 主机编译验证。
// 对应普通版: driver.step_column。

#include <math.h>

namespace {

__device__ __forceinline__ float safe_den(float value) {
    return fabsf(value) < 1.0e-12f ? copysignf(1.0e-12f, value + 1.0e-20f) : value;
}

// 对应普通版: soil._thomas；四层追赶法固定展开。
__device__ void thomas4(const float lower[4], const float diagonal[4],
                        const float upper[4], const float rhs[4], float out[4]) {
    float cp0 = upper[0] / safe_den(diagonal[0]);
    float dp0 = rhs[0] / safe_den(diagonal[0]);
    float b1 = diagonal[1] - lower[1] * cp0;
    float cp1 = upper[1] / safe_den(b1);
    float dp1 = (rhs[1] - lower[1] * dp0) / safe_den(b1);
    float b2 = diagonal[2] - lower[2] * cp1;
    float cp2 = upper[2] / safe_den(b2);
    float dp2 = (rhs[2] - lower[2] * dp1) / safe_den(b2);
    float b3 = diagonal[3] - lower[3] * cp2;
    float dp3 = (rhs[3] - lower[3] * dp2) / safe_den(b3);
    out[3] = dp3;
    out[2] = dp2 - cp2 * out[3];
    out[1] = dp1 - cp1 * out[2];
    out[0] = dp0 - cp0 * out[1];
}

__device__ float energy_residual(float ts, float tair, float rnet0, float rho,
                                 float ce, float soil0, float conductance) {
    float rnet = rnet0 - 5.670374419e-8f * (ts * ts * ts * ts - tair * tair * tair * tair);
    float h = rho * 1004.0f * ce * (ts - tair);
    float g = conductance * (ts - soil0);
    return rnet - h - g;
}

}  // namespace

// state_soa[field * n + column] uses the STATE_SPEC offsets. options_soa has
// 33 integer fields in the same layout. diag_soa[diagnostic * n + column].
extern "C" __global__ void step_columns(
    const float* __restrict__ state_soa,
    const int* __restrict__ options_soa,
    const float* __restrict__ forcing_soa,
    float* __restrict__ state_out_soa,
    int* __restrict__ options_out_soa,
    float* __restrict__ diag_soa,
    int n_columns) {
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int column = first; column < n_columns; column += stride) {
        float tsoil[4];
        float smc[4];
        float lower[4] = {0.0f, -0.2f, -0.2f, -0.2f};
        float diagonal[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        float upper[4] = {-0.2f, -0.2f, -0.2f, 0.0f};
        float rhs[4];
        float solved[4];
        for (int layer = 0; layer < 4; ++layer) {
            tsoil[layer] = state_soa[layer * n_columns + column];
            smc[layer] = state_soa[(4 + layer) * n_columns + column];
            rhs[layer] = tsoil[layer];
        }

        const float dt = forcing_soa[column];
        const float tair = forcing_soa[n_columns + column];
        const float swdown = forcing_soa[5 * n_columns + column];
        const float lwdown = forcing_soa[6 * n_columns + column];
        const float rho = 1.2f;
        const float ce = 0.01f;
        const float rnet0 = 0.8f * swdown + 0.97f * (lwdown - 5.670374419e-8f * tair * tair * tair * tair);

        // Fixed 25-step secant iteration, matching the JAX control-flow contract.
        float x0 = tair;
        float x1 = tair + 1.0f;
        float f0 = energy_residual(x0, tair, rnet0, rho, ce, tsoil[0], 0.4f);
        for (int iteration = 0; iteration < 25; ++iteration) {
            float f1 = energy_residual(x1, tair, rnet0, rho, ce, tsoil[0], 0.4f);
            float denominator = safe_den(f1 - f0);
            float x2 = x1 - f1 * (x1 - x0) / denominator;
            x2 = fminf(330.0f, fmaxf(200.0f, x2));
            x0 = x1;
            f0 = f1;
            x1 = x2;
        }
        const float ts = x1;

        // 固定 4x4 Thomas 热传导骨架；完整模块参数由后续 kernel parity 补齐。
        thomas4(lower, diagonal, upper, rhs, solved);
        for (int layer = 0; layer < 4; ++layer) {
            state_out_soa[layer * n_columns + column] = solved[layer];
            state_out_soa[(4 + layer) * n_columns + column] = smc[layer];
        }
        for (int field = 8; field < 70; ++field) {
            state_out_soa[field * n_columns + column] = state_soa[field * n_columns + column];
        }
        for (int option = 0; option < 33; ++option) {
            options_out_soa[option * n_columns + column] = options_soa[option * n_columns + column];
        }

        const float h = rho * 1004.0f * ce * (ts - tair);
        const float g = 0.4f * (ts - tsoil[0]);
        const float residual = rnet0 - h - g;
        diag_soa[0 * n_columns + column] = ts;
        diag_soa[1 * n_columns + column] = rnet0;
        diag_soa[2 * n_columns + column] = h;
        diag_soa[3 * n_columns + column] = 0.0f;
        diag_soa[4 * n_columns + column] = g;
        diag_soa[5 * n_columns + column] = 0.0f;
        diag_soa[6 * n_columns + column] = fmaxf(0.0f, forcing_soa[7 * n_columns + column]) * dt * 1000.0f;
        diag_soa[7 * n_columns + column] = 0.0f;
        diag_soa[8 * n_columns + column] = 0.0f;
        diag_soa[9 * n_columns + column] = state_soa[26 * n_columns + column];
        diag_soa[10 * n_columns + column] = state_soa[30 * n_columns + column];
        diag_soa[11 * n_columns + column] = residual;
    }
}
