module @jit_geometry_context attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x4x6xf32>, %arg1: tensor<1x12xf32>) -> (tensor<1x29xf32> {jax.result_info = "result[0]"}, tensor<1x4x9xf32> {jax.result_info = "result[1]"}) {
    %0 = stablehlo.slice %arg0 [0:1, 0:4, 0:1] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %cst = stablehlo.constant dense<1.000000e+03> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %3 = stablehlo.divide %1, %2 : tensor<1x4xf32>
    %4 = stablehlo.optimization_barrier %3 : tensor<1x4xf32>
    %5 = stablehlo.slice %arg0 [0:1, 0:4, 1:2] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %6 = stablehlo.reshape %5 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %cst_0 = stablehlo.constant dense<9.170000e+02> : tensor<f32>
    %7 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %8 = stablehlo.divide %6, %7 : tensor<1x4xf32>
    %9 = stablehlo.optimization_barrier %8 : tensor<1x4xf32>
    %10 = stablehlo.add %4, %9 : tensor<1x4xf32>
    %11 = stablehlo.optimization_barrier %10 : tensor<1x4xf32>
    %12 = stablehlo.slice %arg0 [0:1, 0:4, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %13 = stablehlo.reshape %12 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %14 = stablehlo.slice %arg1 [0:1, 0:1] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %15 = stablehlo.reshape %14 : (tensor<1x1xf32>) -> tensor<1xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
    %18 = stablehlo.multiply %13, %17 : tensor<1x4xf32>
    %19 = stablehlo.optimization_barrier %18 : tensor<1x4xf32>
    %20 = stablehlo.divide %11, %19 : tensor<1x4xf32>
    %21 = stablehlo.optimization_barrier %20 : tensor<1x4xf32>
    %22 = stablehlo.slice %arg0 [0:1, 0:4, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %23 = stablehlo.reshape %22 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %cst_1 = stablehlo.constant dense<9.170000e+02> : tensor<f32>
    %24 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %25 = stablehlo.multiply %24, %23 : tensor<1x4xf32>
    %26 = stablehlo.optimization_barrier %25 : tensor<1x4xf32>
    %27 = stablehlo.slice %arg1 [0:1, 0:1] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %28 = stablehlo.reshape %27 : (tensor<1x1xf32>) -> tensor<1xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
    %31 = stablehlo.multiply %26, %30 : tensor<1x4xf32>
    %32 = stablehlo.optimization_barrier %31 : tensor<1x4xf32>
    %33 = stablehlo.slice %arg0 [0:1, 0:4, 1:2] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %34 = stablehlo.reshape %33 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %35 = stablehlo.divide %34, %32 : tensor<1x4xf32>
    %36 = stablehlo.optimization_barrier %35 : tensor<1x4xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %37 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %38 = stablehlo.minimum %37, %36 : tensor<1x4xf32>
    %39 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %40 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %41 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %42 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %43 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %44 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %45 = stablehlo.broadcast_in_dim %32, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %46 = stablehlo.broadcast_in_dim %36, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %47 = stablehlo.broadcast_in_dim %38, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %48 = stablehlo.concatenate %39, %40, %41, %42, %43, %44, %45, %46, %47, dim = 2 : (tensor<1x4x1xf32>, tensor<1x4x1xf32>, tensor<1x4x1xf32>, tensor<1x4x1xf32>, tensor<1x4x1xf32>, tensor<1x4x1xf32>, tensor<1x4x1xf32>, tensor<1x4x1xf32>, tensor<1x4x1xf32>) -> tensor<1x4x9xf32>
    %49 = stablehlo.slice %arg0 [0:1, 0:1, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %50 = stablehlo.reshape %49 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %51 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %52 = stablehlo.add %51, %50 : tensor<1xf32>
    %53 = stablehlo.optimization_barrier %52 : tensor<1xf32>
    %54 = stablehlo.slice %arg0 [0:1, 1:2, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %55 = stablehlo.reshape %54 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %56 = stablehlo.add %53, %55 : tensor<1xf32>
    %57 = stablehlo.optimization_barrier %56 : tensor<1xf32>
    %58 = stablehlo.slice %arg0 [0:1, 2:3, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %59 = stablehlo.reshape %58 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %60 = stablehlo.add %57, %59 : tensor<1xf32>
    %61 = stablehlo.optimization_barrier %60 : tensor<1xf32>
    %62 = stablehlo.slice %arg0 [0:1, 3:4, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %63 = stablehlo.reshape %62 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %64 = stablehlo.add %61, %63 : tensor<1xf32>
    %65 = stablehlo.optimization_barrier %64 : tensor<1xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %66 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %67 = stablehlo.broadcast_in_dim %66, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %68 = stablehlo.broadcast_in_dim %53, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %69 = stablehlo.broadcast_in_dim %57, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %70 = stablehlo.broadcast_in_dim %61, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %71 = stablehlo.broadcast_in_dim %65, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %72 = stablehlo.concatenate %67, %68, %69, %70, %71, dim = 1 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x5xf32>
    %73 = stablehlo.slice %72 [0:1, 0:4] : (tensor<1x5xf32>) -> tensor<1x4xf32>
    %74 = stablehlo.slice %arg0 [0:1, 0:4, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %75 = stablehlo.reshape %74 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %cst_5 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %77 = stablehlo.multiply %76, %75 : tensor<1x4xf32>
    %78 = stablehlo.optimization_barrier %77 : tensor<1x4xf32>
    %79 = stablehlo.add %73, %78 : tensor<1x4xf32>
    %80 = stablehlo.optimization_barrier %79 : tensor<1x4xf32>
    %81 = stablehlo.slice %72 [0:1, 1:5] : (tensor<1x5xf32>) -> tensor<1x4xf32>
    %82 = stablehlo.slice %arg1 [0:1, 4:5] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %83 = stablehlo.reshape %82 : (tensor<1x1xf32>) -> tensor<1xf32>
    %84 = stablehlo.slice %72 [0:1, 4:5] : (tensor<1x5xf32>) -> tensor<1x1xf32>
    %85 = stablehlo.reshape %84 : (tensor<1x1xf32>) -> tensor<1xf32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %86 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %87 = stablehlo.maximum %85, %86 : tensor<1xf32>
    %cst_7 = stablehlo.constant dense<1.90734863E-6> : tensor<f32>
    %88 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %89 = stablehlo.multiply %88, %87 : tensor<1xf32>
    %90 = stablehlo.add %83, %89 : tensor<1xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
    %93 = stablehlo.compare LE, %81, %92, FLOAT : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xi1>
    %94 = stablehlo.convert %93 : (tensor<1x4xi1>) -> tensor<1x4xi32>
    %95 = stablehlo.convert %94 : (tensor<1x4xi32>) -> tensor<1x4xi64>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %96 = stablehlo.reduce(%95 init: %c) applies stablehlo.add across dimensions = [1] : (tensor<1x4xi64>, tensor<i64>) -> tensor<1xi64>
    %97 = stablehlo.slice %arg0 [0:1, 0:4, 0:1] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %98 = stablehlo.reshape %97 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+03> : tensor<f32>
    %99 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %100 = stablehlo.divide %98, %99 : tensor<1x4xf32>
    %101 = stablehlo.optimization_barrier %100 : tensor<1x4xf32>
    %102 = stablehlo.slice %arg0 [0:1, 0:4, 1:2] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %103 = stablehlo.reshape %102 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %cst_9 = stablehlo.constant dense<9.170000e+02> : tensor<f32>
    %104 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %105 = stablehlo.divide %103, %104 : tensor<1x4xf32>
    %106 = stablehlo.optimization_barrier %105 : tensor<1x4xf32>
    %107 = stablehlo.add %101, %106 : tensor<1x4xf32>
    %108 = stablehlo.optimization_barrier %107 : tensor<1x4xf32>
    %109 = stablehlo.slice %arg0 [0:1, 0:4, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %110 = stablehlo.reshape %109 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %111 = stablehlo.slice %arg1 [0:1, 0:1] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %112 = stablehlo.reshape %111 : (tensor<1x1xf32>) -> tensor<1xf32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %114 = stablehlo.broadcast_in_dim %113, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
    %115 = stablehlo.multiply %110, %114 : tensor<1x4xf32>
    %116 = stablehlo.optimization_barrier %115 : tensor<1x4xf32>
    %117 = stablehlo.divide %108, %116 : tensor<1x4xf32>
    %118 = stablehlo.optimization_barrier %117 : tensor<1x4xf32>
    %119 = stablehlo.slice %arg0 [0:1, 0:4, 1:2] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %120 = stablehlo.reshape %119 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %121 = stablehlo.slice %arg0 [0:1, 0:4, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %122 = stablehlo.reshape %121 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %cst_10 = stablehlo.constant dense<9.170000e+02> : tensor<f32>
    %123 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %124 = stablehlo.multiply %123, %122 : tensor<1x4xf32>
    %125 = stablehlo.optimization_barrier %124 : tensor<1x4xf32>
    %126 = stablehlo.slice %arg1 [0:1, 0:1] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %127 = stablehlo.reshape %126 : (tensor<1x1xf32>) -> tensor<1xf32>
    %128 = stablehlo.broadcast_in_dim %127, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %129 = stablehlo.broadcast_in_dim %128, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
    %130 = stablehlo.multiply %125, %129 : tensor<1x4xf32>
    %131 = stablehlo.optimization_barrier %130 : tensor<1x4xf32>
    %132 = stablehlo.divide %120, %131 : tensor<1x4xf32>
    %133 = stablehlo.optimization_barrier %132 : tensor<1x4xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %134 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %135 = stablehlo.minimum %134, %133 : tensor<1x4xf32>
    %136 = stablehlo.slice %arg1 [0:1, 4:5] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %137 = stablehlo.reshape %136 : (tensor<1x1xf32>) -> tensor<1xf32>
    %138 = stablehlo.slice %arg1 [0:1, 8:9] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %139 = stablehlo.reshape %138 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_12 = stablehlo.constant dense<0> : tensor<i64>
    %140 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %141 = stablehlo.compare GT, %96, %140, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %142 = stablehlo.slice %118 [0:1, 0:1] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %143 = stablehlo.reshape %142 : (tensor<1x1xf32>) -> tensor<1xf32>
    %144 = stablehlo.compare LE, %143, %139, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %145 = stablehlo.and %141, %144 : tensor<1xi1>
    %c_13 = stablehlo.constant dense<0> : tensor<i64>
    %c_14 = stablehlo.constant dense<-1> : tensor<i32>
    %146 = call @_where(%145, %c_13, %c_14) : (tensor<1xi1>, tensor<i64>, tensor<i32>) -> tensor<1xi32>
    %c_15 = stablehlo.constant dense<1> : tensor<i64>
    %147 = stablehlo.broadcast_in_dim %c_15, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %148 = stablehlo.compare GT, %96, %147, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %149 = stablehlo.slice %118 [0:1, 1:2] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %150 = stablehlo.reshape %149 : (tensor<1x1xf32>) -> tensor<1xf32>
    %151 = stablehlo.compare LE, %150, %139, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %152 = stablehlo.and %148, %151 : tensor<1xi1>
    %c_16 = stablehlo.constant dense<1> : tensor<i64>
    %153 = call @_where_0(%152, %c_16, %146) : (tensor<1xi1>, tensor<i64>, tensor<1xi32>) -> tensor<1xi32>
    %c_17 = stablehlo.constant dense<2> : tensor<i64>
    %154 = stablehlo.broadcast_in_dim %c_17, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %155 = stablehlo.compare GT, %96, %154, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %156 = stablehlo.slice %118 [0:1, 2:3] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %157 = stablehlo.reshape %156 : (tensor<1x1xf32>) -> tensor<1xf32>
    %158 = stablehlo.compare LE, %157, %139, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %159 = stablehlo.and %155, %158 : tensor<1xi1>
    %c_18 = stablehlo.constant dense<2> : tensor<i64>
    %160 = call @_where_0(%159, %c_18, %153) : (tensor<1xi1>, tensor<i64>, tensor<1xi32>) -> tensor<1xi32>
    %c_19 = stablehlo.constant dense<3> : tensor<i64>
    %161 = stablehlo.broadcast_in_dim %c_19, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %162 = stablehlo.compare GT, %96, %161, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %163 = stablehlo.slice %118 [0:1, 3:4] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %164 = stablehlo.reshape %163 : (tensor<1x1xf32>) -> tensor<1xf32>
    %165 = stablehlo.compare LE, %164, %139, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %166 = stablehlo.and %162, %165 : tensor<1xi1>
    %c_20 = stablehlo.constant dense<3> : tensor<i64>
    %167 = call @_where_0(%166, %c_20, %160) : (tensor<1xi1>, tensor<i64>, tensor<1xi32>) -> tensor<1xi32>
    %c_21 = stablehlo.constant dense<0> : tensor<i64>
    %c_22 = stablehlo.constant dense<2> : tensor<i64>
    %168 = call @clip(%167, %c_21, %c_22) : (tensor<1xi32>, tensor<i64>, tensor<i64>) -> tensor<1xi32>
    %c_23 = stablehlo.constant dense<1> : tensor<i32>
    %169 = stablehlo.broadcast_in_dim %c_23, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %170 = stablehlo.add %168, %169 : tensor<1xi32>
    %c_24 = stablehlo.constant dense<0> : tensor<i32>
    %171 = stablehlo.broadcast_in_dim %c_24, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %172 = stablehlo.compare LT, %170, %171, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_25 = stablehlo.constant dense<4> : tensor<i32>
    %173 = stablehlo.broadcast_in_dim %c_25, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %174 = stablehlo.add %170, %173 : tensor<1xi32>
    %175 = stablehlo.select %172, %174, %170 : tensor<1xi1>, tensor<1xi32>
    %176 = stablehlo.broadcast_in_dim %175, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %177 = "stablehlo.gather"(%118, %176) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %178 = stablehlo.reshape %177 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_26 = stablehlo.constant dense<0> : tensor<i32>
    %179 = stablehlo.broadcast_in_dim %c_26, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %180 = stablehlo.compare LT, %168, %179, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_27 = stablehlo.constant dense<4> : tensor<i32>
    %181 = stablehlo.broadcast_in_dim %c_27, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %182 = stablehlo.add %168, %181 : tensor<1xi32>
    %183 = stablehlo.select %180, %182, %168 : tensor<1xi1>, tensor<1xi32>
    %184 = stablehlo.broadcast_in_dim %183, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %185 = "stablehlo.gather"(%118, %184) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %186 = stablehlo.reshape %185 : (tensor<1x1xf32>) -> tensor<1xf32>
    %187 = stablehlo.subtract %178, %186 : tensor<1xf32>
    %188 = stablehlo.optimization_barrier %187 : tensor<1xf32>
    %c_28 = stablehlo.constant dense<0> : tensor<i32>
    %189 = stablehlo.broadcast_in_dim %c_28, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %190 = stablehlo.compare LT, %168, %189, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_29 = stablehlo.constant dense<4> : tensor<i32>
    %191 = stablehlo.broadcast_in_dim %c_29, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %192 = stablehlo.add %168, %191 : tensor<1xi32>
    %193 = stablehlo.select %190, %192, %168 : tensor<1xi1>, tensor<1xi32>
    %194 = stablehlo.broadcast_in_dim %193, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %195 = "stablehlo.gather"(%118, %194) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %196 = stablehlo.reshape %195 : (tensor<1x1xf32>) -> tensor<1xf32>
    %197 = stablehlo.subtract %139, %196 : tensor<1xf32>
    %198 = stablehlo.optimization_barrier %197 : tensor<1xf32>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %199 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %200 = stablehlo.compare EQ, %188, %199, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %c_31 = stablehlo.constant dense<1> : tensor<i64>
    %201 = call @_where_1(%200, %c_31, %188) : (tensor<1xi1>, tensor<i64>, tensor<1xf32>) -> tensor<1xf32>
    %202 = stablehlo.divide %198, %201 : tensor<1xf32>
    %203 = stablehlo.optimization_barrier %202 : tensor<1xf32>
    %c_32 = stablehlo.constant dense<0> : tensor<i32>
    %204 = stablehlo.broadcast_in_dim %c_32, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %205 = stablehlo.compare LT, %168, %204, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_33 = stablehlo.constant dense<4> : tensor<i32>
    %206 = stablehlo.broadcast_in_dim %c_33, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %207 = stablehlo.add %168, %206 : tensor<1xi32>
    %208 = stablehlo.select %205, %207, %168 : tensor<1xi1>, tensor<1xi32>
    %209 = stablehlo.broadcast_in_dim %208, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %210 = "stablehlo.gather"(%80, %209) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %211 = stablehlo.reshape %210 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_34 = stablehlo.constant dense<1> : tensor<i32>
    %212 = stablehlo.broadcast_in_dim %c_34, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %213 = stablehlo.add %168, %212 : tensor<1xi32>
    %c_35 = stablehlo.constant dense<0> : tensor<i32>
    %214 = stablehlo.broadcast_in_dim %c_35, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %215 = stablehlo.compare LT, %213, %214, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_36 = stablehlo.constant dense<4> : tensor<i32>
    %216 = stablehlo.broadcast_in_dim %c_36, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %217 = stablehlo.add %213, %216 : tensor<1xi32>
    %218 = stablehlo.select %215, %217, %213 : tensor<1xi1>, tensor<1xi32>
    %219 = stablehlo.broadcast_in_dim %218, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %220 = "stablehlo.gather"(%80, %219) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %221 = stablehlo.reshape %220 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_37 = stablehlo.constant dense<0> : tensor<i32>
    %222 = stablehlo.broadcast_in_dim %c_37, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %223 = stablehlo.compare LT, %168, %222, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_38 = stablehlo.constant dense<4> : tensor<i32>
    %224 = stablehlo.broadcast_in_dim %c_38, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %225 = stablehlo.add %168, %224 : tensor<1xi32>
    %226 = stablehlo.select %223, %225, %168 : tensor<1xi1>, tensor<1xi32>
    %227 = stablehlo.broadcast_in_dim %226, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %228 = "stablehlo.gather"(%80, %227) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %229 = stablehlo.reshape %228 : (tensor<1x1xf32>) -> tensor<1xf32>
    %230 = stablehlo.subtract %221, %229 : tensor<1xf32>
    %231 = stablehlo.optimization_barrier %230 : tensor<1xf32>
    %232 = stablehlo.multiply %203, %231 : tensor<1xf32>
    %233 = stablehlo.optimization_barrier %232 : tensor<1xf32>
    %234 = stablehlo.add %211, %233 : tensor<1xf32>
    %235 = stablehlo.optimization_barrier %234 : tensor<1xf32>
    %c_39 = stablehlo.constant dense<0> : tensor<i64>
    %236 = call @clip_2(%235, %c_39, %137) : (tensor<1xf32>, tensor<i64>, tensor<1xf32>) -> tensor<1xf32>
    %c_40 = stablehlo.constant dense<0> : tensor<i32>
    %237 = stablehlo.broadcast_in_dim %c_40, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %238 = stablehlo.compare LE, %167, %237, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %239 = stablehlo.slice %72 [0:1, 1:2] : (tensor<1x5xf32>) -> tensor<1x1xf32>
    %240 = stablehlo.reshape %239 : (tensor<1x1xf32>) -> tensor<1xf32>
    %241 = stablehlo.minimum %240, %137 : tensor<1xf32>
    %c_41 = stablehlo.constant dense<1> : tensor<i64>
    %242 = stablehlo.broadcast_in_dim %c_41, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %243 = stablehlo.subtract %96, %242 : tensor<1xi64>
    %244 = stablehlo.convert %167 : (tensor<1xi32>) -> tensor<1xi64>
    %245 = stablehlo.compare EQ, %244, %243, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %246 = call @_where_3(%245, %137, %236) : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %247 = call @_where_3(%238, %241, %246) : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %c_42 = stablehlo.constant dense<1> : tensor<i64>
    %248 = stablehlo.broadcast_in_dim %c_42, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %249 = stablehlo.compare GT, %96, %248, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %250 = stablehlo.slice %arg0 [0:1, 0:1, 3:4] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %251 = stablehlo.reshape %250 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %cst_43 = stablehlo.constant dense<2.731500e+02> : tensor<f32>
    %252 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %253 = stablehlo.compare GT, %251, %252, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %254 = stablehlo.and %249, %253 : tensor<1xi1>
    %255 = stablehlo.slice %arg0 [0:1, 1:2, 3:4] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %256 = stablehlo.reshape %255 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %cst_44 = stablehlo.constant dense<2.731500e+02> : tensor<f32>
    %257 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %258 = stablehlo.compare LE, %256, %257, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %259 = stablehlo.and %254, %258 : tensor<1xi1>
    %c_45 = stablehlo.constant dense<-1> : tensor<i32>
    %c_46 = stablehlo.constant dense<0> : tensor<i32>
    %260 = stablehlo.compare LT, %c_45, %c_46, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %261 = stablehlo.broadcast_in_dim %260, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %262 = stablehlo.and %259, %261 : tensor<1xi1>
    %c_47 = stablehlo.constant dense<1> : tensor<i64>
    %c_48 = stablehlo.constant dense<-1> : tensor<i32>
    %263 = call @_where(%262, %c_47, %c_48) : (tensor<1xi1>, tensor<i64>, tensor<i32>) -> tensor<1xi32>
    %264 = stablehlo.convert %259 : (tensor<1xi1>) -> tensor<1xi32>
    %c_49 = stablehlo.constant dense<0> : tensor<i32>
    %265 = stablehlo.broadcast_in_dim %c_49, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %266 = stablehlo.add %265, %264 : tensor<1xi32>
    %c_50 = stablehlo.constant dense<2> : tensor<i64>
    %267 = stablehlo.broadcast_in_dim %c_50, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %268 = stablehlo.compare GT, %96, %267, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %269 = stablehlo.slice %arg0 [0:1, 1:2, 3:4] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %270 = stablehlo.reshape %269 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %cst_51 = stablehlo.constant dense<2.731500e+02> : tensor<f32>
    %271 = stablehlo.broadcast_in_dim %cst_51, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %272 = stablehlo.compare GT, %270, %271, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %273 = stablehlo.and %268, %272 : tensor<1xi1>
    %274 = stablehlo.slice %arg0 [0:1, 2:3, 3:4] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %275 = stablehlo.reshape %274 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %cst_52 = stablehlo.constant dense<2.731500e+02> : tensor<f32>
    %276 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %277 = stablehlo.compare LE, %275, %276, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %278 = stablehlo.and %273, %277 : tensor<1xi1>
    %c_53 = stablehlo.constant dense<0> : tensor<i32>
    %279 = stablehlo.broadcast_in_dim %c_53, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %280 = stablehlo.compare LT, %263, %279, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %281 = stablehlo.and %278, %280 : tensor<1xi1>
    %c_54 = stablehlo.constant dense<2> : tensor<i64>
    %282 = call @_where_0(%281, %c_54, %263) : (tensor<1xi1>, tensor<i64>, tensor<1xi32>) -> tensor<1xi32>
    %283 = stablehlo.convert %278 : (tensor<1xi1>) -> tensor<1xi32>
    %284 = stablehlo.add %266, %283 : tensor<1xi32>
    %c_55 = stablehlo.constant dense<3> : tensor<i64>
    %285 = stablehlo.broadcast_in_dim %c_55, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %286 = stablehlo.compare GT, %96, %285, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %287 = stablehlo.slice %arg0 [0:1, 2:3, 3:4] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %288 = stablehlo.reshape %287 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %cst_56 = stablehlo.constant dense<2.731500e+02> : tensor<f32>
    %289 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %290 = stablehlo.compare GT, %288, %289, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %291 = stablehlo.and %286, %290 : tensor<1xi1>
    %292 = stablehlo.slice %arg0 [0:1, 3:4, 3:4] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %293 = stablehlo.reshape %292 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %cst_57 = stablehlo.constant dense<2.731500e+02> : tensor<f32>
    %294 = stablehlo.broadcast_in_dim %cst_57, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %295 = stablehlo.compare LE, %293, %294, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %296 = stablehlo.and %291, %295 : tensor<1xi1>
    %c_58 = stablehlo.constant dense<0> : tensor<i32>
    %297 = stablehlo.broadcast_in_dim %c_58, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %298 = stablehlo.compare LT, %282, %297, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %299 = stablehlo.and %296, %298 : tensor<1xi1>
    %c_59 = stablehlo.constant dense<3> : tensor<i64>
    %300 = call @_where_0(%299, %c_59, %282) : (tensor<1xi1>, tensor<i64>, tensor<1xi32>) -> tensor<1xi32>
    %301 = stablehlo.convert %296 : (tensor<1xi1>) -> tensor<1xi32>
    %302 = stablehlo.add %284, %301 : tensor<1xi32>
    %c_60 = stablehlo.constant dense<0> : tensor<i32>
    %303 = stablehlo.broadcast_in_dim %c_60, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %304 = stablehlo.compare GE, %300, %303, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_61 = stablehlo.constant dense<0> : tensor<i32>
    %305 = stablehlo.broadcast_in_dim %c_61, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %306 = stablehlo.maximum %300, %305 : tensor<1xi32>
    %c_62 = stablehlo.constant dense<0> : tensor<i32>
    %307 = stablehlo.broadcast_in_dim %c_62, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %308 = stablehlo.compare LT, %306, %307, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_63 = stablehlo.constant dense<4> : tensor<i32>
    %309 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %310 = stablehlo.add %306, %309 : tensor<1xi32>
    %311 = stablehlo.select %308, %310, %306 : tensor<1xi1>, tensor<1xi32>
    %312 = stablehlo.broadcast_in_dim %311, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %313 = "stablehlo.gather"(%80, %312) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %314 = stablehlo.reshape %313 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_64 = stablehlo.constant dense<0> : tensor<i64>
    %315 = call @_where_4(%304, %314, %c_64) : (tensor<1xi1>, tensor<1xf32>, tensor<i64>) -> tensor<1xf32>
    %c_65 = stablehlo.constant dense<1> : tensor<i32>
    %316 = stablehlo.broadcast_in_dim %c_65, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %317 = stablehlo.add %300, %316 : tensor<1xi32>
    %318 = stablehlo.slice %arg1 [0:1, 8:9] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %319 = stablehlo.reshape %318 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_66 = stablehlo.constant dense<0> : tensor<i32>
    %320 = stablehlo.broadcast_in_dim %c_66, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %321 = stablehlo.compare GT, %317, %320, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %322 = stablehlo.slice %118 [0:1, 0:1] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %323 = stablehlo.reshape %322 : (tensor<1x1xf32>) -> tensor<1xf32>
    %324 = stablehlo.compare LE, %323, %319, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %325 = stablehlo.and %321, %324 : tensor<1xi1>
    %c_67 = stablehlo.constant dense<0> : tensor<i64>
    %c_68 = stablehlo.constant dense<-1> : tensor<i32>
    %326 = call @_where(%325, %c_67, %c_68) : (tensor<1xi1>, tensor<i64>, tensor<i32>) -> tensor<1xi32>
    %c_69 = stablehlo.constant dense<1> : tensor<i32>
    %327 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %328 = stablehlo.compare GT, %317, %327, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %329 = stablehlo.slice %118 [0:1, 1:2] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %330 = stablehlo.reshape %329 : (tensor<1x1xf32>) -> tensor<1xf32>
    %331 = stablehlo.compare LE, %330, %319, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %332 = stablehlo.and %328, %331 : tensor<1xi1>
    %c_70 = stablehlo.constant dense<1> : tensor<i64>
    %333 = call @_where_0(%332, %c_70, %326) : (tensor<1xi1>, tensor<i64>, tensor<1xi32>) -> tensor<1xi32>
    %c_71 = stablehlo.constant dense<2> : tensor<i32>
    %334 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %335 = stablehlo.compare GT, %317, %334, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %336 = stablehlo.slice %118 [0:1, 2:3] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %337 = stablehlo.reshape %336 : (tensor<1x1xf32>) -> tensor<1xf32>
    %338 = stablehlo.compare LE, %337, %319, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %339 = stablehlo.and %335, %338 : tensor<1xi1>
    %c_72 = stablehlo.constant dense<2> : tensor<i64>
    %340 = call @_where_0(%339, %c_72, %333) : (tensor<1xi1>, tensor<i64>, tensor<1xi32>) -> tensor<1xi32>
    %c_73 = stablehlo.constant dense<3> : tensor<i32>
    %341 = stablehlo.broadcast_in_dim %c_73, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %342 = stablehlo.compare GT, %317, %341, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %343 = stablehlo.slice %118 [0:1, 3:4] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %344 = stablehlo.reshape %343 : (tensor<1x1xf32>) -> tensor<1xf32>
    %345 = stablehlo.compare LE, %344, %319, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %346 = stablehlo.and %342, %345 : tensor<1xi1>
    %c_74 = stablehlo.constant dense<3> : tensor<i64>
    %347 = call @_where_0(%346, %c_74, %340) : (tensor<1xi1>, tensor<i64>, tensor<1xi32>) -> tensor<1xi32>
    %c_75 = stablehlo.constant dense<0> : tensor<i64>
    %c_76 = stablehlo.constant dense<2> : tensor<i64>
    %348 = call @clip(%347, %c_75, %c_76) : (tensor<1xi32>, tensor<i64>, tensor<i64>) -> tensor<1xi32>
    %c_77 = stablehlo.constant dense<1> : tensor<i32>
    %349 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %350 = stablehlo.add %348, %349 : tensor<1xi32>
    %c_78 = stablehlo.constant dense<0> : tensor<i32>
    %351 = stablehlo.broadcast_in_dim %c_78, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %352 = stablehlo.compare LT, %350, %351, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_79 = stablehlo.constant dense<4> : tensor<i32>
    %353 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %354 = stablehlo.add %350, %353 : tensor<1xi32>
    %355 = stablehlo.select %352, %354, %350 : tensor<1xi1>, tensor<1xi32>
    %356 = stablehlo.broadcast_in_dim %355, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %357 = "stablehlo.gather"(%118, %356) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %358 = stablehlo.reshape %357 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_80 = stablehlo.constant dense<0> : tensor<i32>
    %359 = stablehlo.broadcast_in_dim %c_80, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %360 = stablehlo.compare LT, %348, %359, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_81 = stablehlo.constant dense<4> : tensor<i32>
    %361 = stablehlo.broadcast_in_dim %c_81, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %362 = stablehlo.add %348, %361 : tensor<1xi32>
    %363 = stablehlo.select %360, %362, %348 : tensor<1xi1>, tensor<1xi32>
    %364 = stablehlo.broadcast_in_dim %363, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %365 = "stablehlo.gather"(%118, %364) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %366 = stablehlo.reshape %365 : (tensor<1x1xf32>) -> tensor<1xf32>
    %367 = stablehlo.subtract %358, %366 : tensor<1xf32>
    %368 = stablehlo.optimization_barrier %367 : tensor<1xf32>
    %c_82 = stablehlo.constant dense<0> : tensor<i32>
    %369 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %370 = stablehlo.compare LT, %348, %369, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_83 = stablehlo.constant dense<4> : tensor<i32>
    %371 = stablehlo.broadcast_in_dim %c_83, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %372 = stablehlo.add %348, %371 : tensor<1xi32>
    %373 = stablehlo.select %370, %372, %348 : tensor<1xi1>, tensor<1xi32>
    %374 = stablehlo.broadcast_in_dim %373, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %375 = "stablehlo.gather"(%118, %374) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %376 = stablehlo.reshape %375 : (tensor<1x1xf32>) -> tensor<1xf32>
    %377 = stablehlo.subtract %319, %376 : tensor<1xf32>
    %378 = stablehlo.optimization_barrier %377 : tensor<1xf32>
    %cst_84 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %379 = stablehlo.broadcast_in_dim %cst_84, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %380 = stablehlo.compare EQ, %368, %379, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %c_85 = stablehlo.constant dense<1> : tensor<i64>
    %381 = call @_where_1(%380, %c_85, %368) : (tensor<1xi1>, tensor<i64>, tensor<1xf32>) -> tensor<1xf32>
    %382 = stablehlo.divide %378, %381 : tensor<1xf32>
    %383 = stablehlo.optimization_barrier %382 : tensor<1xf32>
    %c_86 = stablehlo.constant dense<0> : tensor<i32>
    %384 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %385 = stablehlo.compare LT, %348, %384, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_87 = stablehlo.constant dense<4> : tensor<i32>
    %386 = stablehlo.broadcast_in_dim %c_87, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %387 = stablehlo.add %348, %386 : tensor<1xi32>
    %388 = stablehlo.select %385, %387, %348 : tensor<1xi1>, tensor<1xi32>
    %389 = stablehlo.broadcast_in_dim %388, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %390 = "stablehlo.gather"(%80, %389) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %391 = stablehlo.reshape %390 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_88 = stablehlo.constant dense<1> : tensor<i32>
    %392 = stablehlo.broadcast_in_dim %c_88, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %393 = stablehlo.add %348, %392 : tensor<1xi32>
    %c_89 = stablehlo.constant dense<0> : tensor<i32>
    %394 = stablehlo.broadcast_in_dim %c_89, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %395 = stablehlo.compare LT, %393, %394, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_90 = stablehlo.constant dense<4> : tensor<i32>
    %396 = stablehlo.broadcast_in_dim %c_90, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %397 = stablehlo.add %393, %396 : tensor<1xi32>
    %398 = stablehlo.select %395, %397, %393 : tensor<1xi1>, tensor<1xi32>
    %399 = stablehlo.broadcast_in_dim %398, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %400 = "stablehlo.gather"(%80, %399) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %401 = stablehlo.reshape %400 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_91 = stablehlo.constant dense<0> : tensor<i32>
    %402 = stablehlo.broadcast_in_dim %c_91, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %403 = stablehlo.compare LT, %348, %402, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_92 = stablehlo.constant dense<4> : tensor<i32>
    %404 = stablehlo.broadcast_in_dim %c_92, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %405 = stablehlo.add %348, %404 : tensor<1xi32>
    %406 = stablehlo.select %403, %405, %348 : tensor<1xi1>, tensor<1xi32>
    %407 = stablehlo.broadcast_in_dim %406, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %408 = "stablehlo.gather"(%80, %407) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %409 = stablehlo.reshape %408 : (tensor<1x1xf32>) -> tensor<1xf32>
    %410 = stablehlo.subtract %401, %409 : tensor<1xf32>
    %411 = stablehlo.optimization_barrier %410 : tensor<1xf32>
    %412 = stablehlo.multiply %383, %411 : tensor<1xf32>
    %413 = stablehlo.optimization_barrier %412 : tensor<1xf32>
    %414 = stablehlo.add %391, %413 : tensor<1xf32>
    %415 = stablehlo.optimization_barrier %414 : tensor<1xf32>
    %c_93 = stablehlo.constant dense<0> : tensor<i64>
    %416 = call @clip_2(%415, %c_93, %315) : (tensor<1xf32>, tensor<i64>, tensor<1xf32>) -> tensor<1xf32>
    %c_94 = stablehlo.constant dense<0> : tensor<i32>
    %417 = stablehlo.broadcast_in_dim %c_94, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %418 = stablehlo.compare LT, %347, %417, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_95 = stablehlo.constant dense<1> : tensor<i32>
    %419 = stablehlo.broadcast_in_dim %c_95, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %420 = stablehlo.subtract %317, %419 : tensor<1xi32>
    %421 = stablehlo.compare EQ, %347, %420, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %422 = call @_where_3(%421, %315, %416) : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %c_96 = stablehlo.constant dense<0> : tensor<i64>
    %423 = call @_where_1(%418, %c_96, %422) : (tensor<1xi1>, tensor<i64>, tensor<1xf32>) -> tensor<1xf32>
    %c_97 = stablehlo.constant dense<0> : tensor<i32>
    %424 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %425 = stablehlo.compare GE, %300, %424, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %426 = stablehlo.compare GE, %247, %315, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %427 = stablehlo.and %425, %426 : tensor<1xi1>
    %428 = stablehlo.compare LT, %423, %315, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %429 = stablehlo.and %427, %428 : tensor<1xi1>
    %c_98 = stablehlo.constant dense<0> : tensor<i64>
    %430 = call @_where_4(%429, %423, %c_98) : (tensor<1xi1>, tensor<1xf32>, tensor<i64>) -> tensor<1xf32>
    %431 = stablehlo.slice %arg1 [0:1, 4:5] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %432 = stablehlo.reshape %431 : (tensor<1x1xf32>) -> tensor<1xf32>
    %433 = stablehlo.slice %72 [0:1, 1:5] : (tensor<1x5xf32>) -> tensor<1x4xf32>
    %434 = stablehlo.broadcast_in_dim %247, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %435 = stablehlo.broadcast_in_dim %434, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
    %436 = stablehlo.compare LT, %433, %435, FLOAT : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xi1>
    %437 = stablehlo.convert %436 : (tensor<1x4xi1>) -> tensor<1x4xi32>
    %438 = stablehlo.convert %437 : (tensor<1x4xi32>) -> tensor<1x4xi64>
    %c_99 = stablehlo.constant dense<0> : tensor<i64>
    %439 = stablehlo.reduce(%438 init: %c_99) applies stablehlo.add across dimensions = [1] : (tensor<1x4xi64>, tensor<i64>) -> tensor<1xi64>
    %440 = stablehlo.subtract %432, %247 : tensor<1xf32>
    %441 = stablehlo.optimization_barrier %440 : tensor<1xf32>
    %c_100 = stablehlo.constant dense<1> : tensor<i64>
    %442 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %443 = stablehlo.subtract %439, %442 : tensor<1xi64>
    %c_101 = stablehlo.constant dense<0> : tensor<i64>
    %444 = stablehlo.broadcast_in_dim %c_101, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %445 = stablehlo.maximum %443, %444 : tensor<1xi64>
    %c_102 = stablehlo.constant dense<0> : tensor<i64>
    %446 = stablehlo.broadcast_in_dim %c_102, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %447 = stablehlo.compare LE, %445, %446, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %448 = stablehlo.slice %arg0 [0:1, 0:1, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %449 = stablehlo.reshape %448 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_103 = stablehlo.constant dense<0> : tensor<i64>
    %450 = call @_where_4(%447, %449, %c_103) : (tensor<1xi1>, tensor<1xf32>, tensor<i64>) -> tensor<1xf32>
    %451 = stablehlo.slice %135 [0:1, 0:1] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %452 = stablehlo.reshape %451 : (tensor<1x1xf32>) -> tensor<1xf32>
    %453 = stablehlo.multiply %450, %452 : tensor<1xf32>
    %454 = stablehlo.optimization_barrier %453 : tensor<1xf32>
    %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %455 = stablehlo.broadcast_in_dim %cst_104, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %456 = stablehlo.add %455, %454 : tensor<1xf32>
    %457 = stablehlo.optimization_barrier %456 : tensor<1xf32>
    %cst_105 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %458 = stablehlo.broadcast_in_dim %cst_105, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %459 = stablehlo.add %458, %450 : tensor<1xf32>
    %460 = stablehlo.optimization_barrier %459 : tensor<1xf32>
    %c_106 = stablehlo.constant dense<1> : tensor<i64>
    %461 = stablehlo.broadcast_in_dim %c_106, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %462 = stablehlo.subtract %439, %461 : tensor<1xi64>
    %c_107 = stablehlo.constant dense<0> : tensor<i64>
    %463 = stablehlo.broadcast_in_dim %c_107, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %464 = stablehlo.maximum %462, %463 : tensor<1xi64>
    %c_108 = stablehlo.constant dense<1> : tensor<i64>
    %465 = stablehlo.broadcast_in_dim %c_108, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %466 = stablehlo.compare LE, %464, %465, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %467 = stablehlo.slice %arg0 [0:1, 1:2, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %468 = stablehlo.reshape %467 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_109 = stablehlo.constant dense<0> : tensor<i64>
    %469 = call @_where_4(%466, %468, %c_109) : (tensor<1xi1>, tensor<1xf32>, tensor<i64>) -> tensor<1xf32>
    %470 = stablehlo.slice %135 [0:1, 1:2] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %471 = stablehlo.reshape %470 : (tensor<1x1xf32>) -> tensor<1xf32>
    %472 = stablehlo.multiply %469, %471 : tensor<1xf32>
    %473 = stablehlo.optimization_barrier %472 : tensor<1xf32>
    %474 = stablehlo.add %457, %473 : tensor<1xf32>
    %475 = stablehlo.optimization_barrier %474 : tensor<1xf32>
    %476 = stablehlo.add %460, %469 : tensor<1xf32>
    %477 = stablehlo.optimization_barrier %476 : tensor<1xf32>
    %c_110 = stablehlo.constant dense<1> : tensor<i64>
    %478 = stablehlo.broadcast_in_dim %c_110, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %479 = stablehlo.subtract %439, %478 : tensor<1xi64>
    %c_111 = stablehlo.constant dense<0> : tensor<i64>
    %480 = stablehlo.broadcast_in_dim %c_111, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %481 = stablehlo.maximum %479, %480 : tensor<1xi64>
    %c_112 = stablehlo.constant dense<2> : tensor<i64>
    %482 = stablehlo.broadcast_in_dim %c_112, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %483 = stablehlo.compare LE, %481, %482, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %484 = stablehlo.slice %arg0 [0:1, 2:3, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %485 = stablehlo.reshape %484 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_113 = stablehlo.constant dense<0> : tensor<i64>
    %486 = call @_where_4(%483, %485, %c_113) : (tensor<1xi1>, tensor<1xf32>, tensor<i64>) -> tensor<1xf32>
    %487 = stablehlo.slice %135 [0:1, 2:3] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %488 = stablehlo.reshape %487 : (tensor<1x1xf32>) -> tensor<1xf32>
    %489 = stablehlo.multiply %486, %488 : tensor<1xf32>
    %490 = stablehlo.optimization_barrier %489 : tensor<1xf32>
    %491 = stablehlo.add %475, %490 : tensor<1xf32>
    %492 = stablehlo.optimization_barrier %491 : tensor<1xf32>
    %493 = stablehlo.add %477, %486 : tensor<1xf32>
    %494 = stablehlo.optimization_barrier %493 : tensor<1xf32>
    %c_114 = stablehlo.constant dense<1> : tensor<i64>
    %495 = stablehlo.broadcast_in_dim %c_114, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %496 = stablehlo.subtract %439, %495 : tensor<1xi64>
    %c_115 = stablehlo.constant dense<0> : tensor<i64>
    %497 = stablehlo.broadcast_in_dim %c_115, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %498 = stablehlo.maximum %496, %497 : tensor<1xi64>
    %c_116 = stablehlo.constant dense<3> : tensor<i64>
    %499 = stablehlo.broadcast_in_dim %c_116, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %500 = stablehlo.compare LE, %498, %499, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %501 = stablehlo.slice %arg0 [0:1, 3:4, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %502 = stablehlo.reshape %501 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_117 = stablehlo.constant dense<0> : tensor<i64>
    %503 = call @_where_4(%500, %502, %c_117) : (tensor<1xi1>, tensor<1xf32>, tensor<i64>) -> tensor<1xf32>
    %504 = stablehlo.slice %135 [0:1, 3:4] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %505 = stablehlo.reshape %504 : (tensor<1x1xf32>) -> tensor<1xf32>
    %506 = stablehlo.multiply %503, %505 : tensor<1xf32>
    %507 = stablehlo.optimization_barrier %506 : tensor<1xf32>
    %508 = stablehlo.add %492, %507 : tensor<1xf32>
    %509 = stablehlo.optimization_barrier %508 : tensor<1xf32>
    %510 = stablehlo.add %494, %503 : tensor<1xf32>
    %511 = stablehlo.optimization_barrier %510 : tensor<1xf32>
    %cst_118 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %512 = stablehlo.broadcast_in_dim %cst_118, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %513 = stablehlo.compare EQ, %511, %512, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %c_119 = stablehlo.constant dense<1> : tensor<i64>
    %514 = call @_where_1(%513, %c_119, %511) : (tensor<1xi1>, tensor<i64>, tensor<1xf32>) -> tensor<1xf32>
    %515 = stablehlo.divide %509, %514 : tensor<1xf32>
    %516 = stablehlo.optimization_barrier %515 : tensor<1xf32>
    %517 = stablehlo.slice %arg1 [0:1, 11:12] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %518 = stablehlo.reshape %517 : (tensor<1x1xf32>) -> tensor<1xf32>
    %519 = stablehlo.negate %518 : tensor<1xf32>
    %520 = stablehlo.multiply %519, %516 : tensor<1xf32>
    %521 = stablehlo.optimization_barrier %520 : tensor<1xf32>
    %cst_120 = stablehlo.constant dense<1.000000e+01> : tensor<f32>
    %522 = stablehlo.broadcast_in_dim %cst_120, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %523 = stablehlo.power %522, %521 : tensor<1xf32>
    %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %524 = stablehlo.broadcast_in_dim %cst_121, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %525 = stablehlo.compare EQ, %441, %524, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %c_122 = stablehlo.constant dense<1> : tensor<i64>
    %526 = call @_where_1(%525, %c_122, %523) : (tensor<1xi1>, tensor<i64>, tensor<1xf32>) -> tensor<1xf32>
    %527 = stablehlo.slice %arg1 [0:1, 6:7] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %528 = stablehlo.reshape %527 : (tensor<1x1xf32>) -> tensor<1xf32>
    %529 = stablehlo.slice %arg1 [0:1, 5:6] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %530 = stablehlo.reshape %529 : (tensor<1x1xf32>) -> tensor<1xf32>
    %531 = stablehlo.tan %530 : tensor<1xf32>
    %532 = stablehlo.multiply %528, %531 : tensor<1xf32>
    %533 = stablehlo.optimization_barrier %532 : tensor<1xf32>
    %534 = stablehlo.multiply %533, %441 : tensor<1xf32>
    %535 = stablehlo.optimization_barrier %534 : tensor<1xf32>
    %536 = stablehlo.multiply %535, %526 : tensor<1xf32>
    %537 = stablehlo.optimization_barrier %536 : tensor<1xf32>
    %538 = stablehlo.broadcast_in_dim %450, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %539 = stablehlo.broadcast_in_dim %469, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %540 = stablehlo.broadcast_in_dim %486, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %541 = stablehlo.broadcast_in_dim %503, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %542 = stablehlo.concatenate %538, %539, %540, %541, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<4x1xf32>
    %543 = stablehlo.slice %72 [0:1, 1:5] : (tensor<1x5xf32>) -> tensor<1x4xf32>
    %544 = stablehlo.broadcast_in_dim %430, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %545 = stablehlo.broadcast_in_dim %544, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
    %546 = stablehlo.compare LT, %543, %545, FLOAT : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xi1>
    %547 = stablehlo.convert %546 : (tensor<1x4xi1>) -> tensor<1x4xi32>
    %548 = stablehlo.convert %547 : (tensor<1x4xi32>) -> tensor<1x4xi64>
    %c_123 = stablehlo.constant dense<0> : tensor<i64>
    %549 = stablehlo.reduce(%548 init: %c_123) applies stablehlo.add across dimensions = [1] : (tensor<1x4xi64>, tensor<i64>) -> tensor<1xi64>
    %550 = stablehlo.slice %72 [0:1, 1:5] : (tensor<1x5xf32>) -> tensor<1x4xf32>
    %551 = stablehlo.broadcast_in_dim %315, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %552 = stablehlo.broadcast_in_dim %551, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
    %553 = stablehlo.compare LT, %550, %552, FLOAT : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xi1>
    %554 = stablehlo.convert %553 : (tensor<1x4xi1>) -> tensor<1x4xi32>
    %555 = stablehlo.convert %554 : (tensor<1x4xi32>) -> tensor<1x4xi64>
    %c_124 = stablehlo.constant dense<0> : tensor<i64>
    %556 = stablehlo.reduce(%555 init: %c_124) applies stablehlo.add across dimensions = [1] : (tensor<1x4xi64>, tensor<i64>) -> tensor<1xi64>
    %557 = stablehlo.subtract %315, %430 : tensor<1xf32>
    %558 = stablehlo.optimization_barrier %557 : tensor<1xf32>
    %c_125 = stablehlo.constant dense<0> : tensor<i64>
    %559 = stablehlo.broadcast_in_dim %c_125, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %560 = stablehlo.compare LE, %549, %559, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %c_126 = stablehlo.constant dense<0> : tensor<i64>
    %561 = stablehlo.broadcast_in_dim %c_126, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %562 = stablehlo.compare GE, %556, %561, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %563 = stablehlo.and %560, %562 : tensor<1xi1>
    %564 = stablehlo.slice %arg0 [0:1, 0:1, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %565 = stablehlo.reshape %564 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_127 = stablehlo.constant dense<0> : tensor<i64>
    %566 = call @_where_4(%563, %565, %c_127) : (tensor<1xi1>, tensor<1xf32>, tensor<i64>) -> tensor<1xf32>
    %567 = stablehlo.slice %arg1 [0:1, 11:12] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %568 = stablehlo.reshape %567 : (tensor<1x1xf32>) -> tensor<1xf32>
    %569 = stablehlo.negate %568 : tensor<1xf32>
    %570 = stablehlo.slice %135 [0:1, 0:1] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %571 = stablehlo.reshape %570 : (tensor<1x1xf32>) -> tensor<1xf32>
    %572 = stablehlo.slice %135 [0:1, 1:2] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %573 = stablehlo.reshape %572 : (tensor<1x1xf32>) -> tensor<1xf32>
    %574 = stablehlo.add %571, %573 : tensor<1xf32>
    %575 = stablehlo.optimization_barrier %574 : tensor<1xf32>
    %cst_128 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %576 = stablehlo.broadcast_in_dim %cst_128, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %577 = stablehlo.multiply %576, %575 : tensor<1xf32>
    %578 = stablehlo.optimization_barrier %577 : tensor<1xf32>
    %579 = stablehlo.multiply %569, %578 : tensor<1xf32>
    %580 = stablehlo.optimization_barrier %579 : tensor<1xf32>
    %cst_129 = stablehlo.constant dense<1.000000e+01> : tensor<f32>
    %581 = stablehlo.broadcast_in_dim %cst_129, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %582 = stablehlo.power %581, %580 : tensor<1xf32>
    %583 = stablehlo.multiply %566, %582 : tensor<1xf32>
    %584 = stablehlo.optimization_barrier %583 : tensor<1xf32>
    %cst_130 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %585 = stablehlo.broadcast_in_dim %cst_130, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %586 = stablehlo.add %585, %584 : tensor<1xf32>
    %587 = stablehlo.optimization_barrier %586 : tensor<1xf32>
    %cst_131 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %588 = stablehlo.broadcast_in_dim %cst_131, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %589 = stablehlo.add %588, %566 : tensor<1xf32>
    %590 = stablehlo.optimization_barrier %589 : tensor<1xf32>
    %c_132 = stablehlo.constant dense<1> : tensor<i64>
    %591 = stablehlo.broadcast_in_dim %c_132, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %592 = stablehlo.compare LE, %549, %591, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %c_133 = stablehlo.constant dense<1> : tensor<i64>
    %593 = stablehlo.broadcast_in_dim %c_133, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %594 = stablehlo.compare GE, %556, %593, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %595 = stablehlo.and %592, %594 : tensor<1xi1>
    %596 = stablehlo.slice %arg0 [0:1, 1:2, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %597 = stablehlo.reshape %596 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_134 = stablehlo.constant dense<0> : tensor<i64>
    %598 = call @_where_4(%595, %597, %c_134) : (tensor<1xi1>, tensor<1xf32>, tensor<i64>) -> tensor<1xf32>
    %599 = stablehlo.slice %arg1 [0:1, 11:12] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %600 = stablehlo.reshape %599 : (tensor<1x1xf32>) -> tensor<1xf32>
    %601 = stablehlo.negate %600 : tensor<1xf32>
    %602 = stablehlo.slice %135 [0:1, 1:2] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %603 = stablehlo.reshape %602 : (tensor<1x1xf32>) -> tensor<1xf32>
    %604 = stablehlo.slice %135 [0:1, 2:3] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %605 = stablehlo.reshape %604 : (tensor<1x1xf32>) -> tensor<1xf32>
    %606 = stablehlo.add %603, %605 : tensor<1xf32>
    %607 = stablehlo.optimization_barrier %606 : tensor<1xf32>
    %cst_135 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %608 = stablehlo.broadcast_in_dim %cst_135, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %609 = stablehlo.multiply %608, %607 : tensor<1xf32>
    %610 = stablehlo.optimization_barrier %609 : tensor<1xf32>
    %611 = stablehlo.multiply %601, %610 : tensor<1xf32>
    %612 = stablehlo.optimization_barrier %611 : tensor<1xf32>
    %cst_136 = stablehlo.constant dense<1.000000e+01> : tensor<f32>
    %613 = stablehlo.broadcast_in_dim %cst_136, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %614 = stablehlo.power %613, %612 : tensor<1xf32>
    %615 = stablehlo.multiply %598, %614 : tensor<1xf32>
    %616 = stablehlo.optimization_barrier %615 : tensor<1xf32>
    %617 = stablehlo.add %587, %616 : tensor<1xf32>
    %618 = stablehlo.optimization_barrier %617 : tensor<1xf32>
    %619 = stablehlo.add %590, %598 : tensor<1xf32>
    %620 = stablehlo.optimization_barrier %619 : tensor<1xf32>
    %c_137 = stablehlo.constant dense<2> : tensor<i64>
    %621 = stablehlo.broadcast_in_dim %c_137, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %622 = stablehlo.compare LE, %549, %621, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %c_138 = stablehlo.constant dense<2> : tensor<i64>
    %623 = stablehlo.broadcast_in_dim %c_138, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %624 = stablehlo.compare GE, %556, %623, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %625 = stablehlo.and %622, %624 : tensor<1xi1>
    %626 = stablehlo.slice %arg0 [0:1, 2:3, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %627 = stablehlo.reshape %626 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_139 = stablehlo.constant dense<0> : tensor<i64>
    %628 = call @_where_4(%625, %627, %c_139) : (tensor<1xi1>, tensor<1xf32>, tensor<i64>) -> tensor<1xf32>
    %629 = stablehlo.slice %arg1 [0:1, 11:12] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %630 = stablehlo.reshape %629 : (tensor<1x1xf32>) -> tensor<1xf32>
    %631 = stablehlo.negate %630 : tensor<1xf32>
    %632 = stablehlo.slice %135 [0:1, 2:3] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %633 = stablehlo.reshape %632 : (tensor<1x1xf32>) -> tensor<1xf32>
    %634 = stablehlo.slice %135 [0:1, 3:4] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %635 = stablehlo.reshape %634 : (tensor<1x1xf32>) -> tensor<1xf32>
    %636 = stablehlo.add %633, %635 : tensor<1xf32>
    %637 = stablehlo.optimization_barrier %636 : tensor<1xf32>
    %cst_140 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %638 = stablehlo.broadcast_in_dim %cst_140, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %639 = stablehlo.multiply %638, %637 : tensor<1xf32>
    %640 = stablehlo.optimization_barrier %639 : tensor<1xf32>
    %641 = stablehlo.multiply %631, %640 : tensor<1xf32>
    %642 = stablehlo.optimization_barrier %641 : tensor<1xf32>
    %cst_141 = stablehlo.constant dense<1.000000e+01> : tensor<f32>
    %643 = stablehlo.broadcast_in_dim %cst_141, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %644 = stablehlo.power %643, %642 : tensor<1xf32>
    %645 = stablehlo.multiply %628, %644 : tensor<1xf32>
    %646 = stablehlo.optimization_barrier %645 : tensor<1xf32>
    %647 = stablehlo.add %618, %646 : tensor<1xf32>
    %648 = stablehlo.optimization_barrier %647 : tensor<1xf32>
    %649 = stablehlo.add %620, %628 : tensor<1xf32>
    %650 = stablehlo.optimization_barrier %649 : tensor<1xf32>
    %c_142 = stablehlo.constant dense<3> : tensor<i64>
    %651 = stablehlo.broadcast_in_dim %c_142, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %652 = stablehlo.compare LE, %549, %651, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %c_143 = stablehlo.constant dense<3> : tensor<i64>
    %653 = stablehlo.broadcast_in_dim %c_143, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %654 = stablehlo.compare GE, %556, %653, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %655 = stablehlo.and %652, %654 : tensor<1xi1>
    %656 = stablehlo.slice %arg0 [0:1, 3:4, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %657 = stablehlo.reshape %656 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_144 = stablehlo.constant dense<0> : tensor<i64>
    %658 = call @_where_4(%655, %657, %c_144) : (tensor<1xi1>, tensor<1xf32>, tensor<i64>) -> tensor<1xf32>
    %659 = stablehlo.slice %arg1 [0:1, 11:12] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %660 = stablehlo.reshape %659 : (tensor<1x1xf32>) -> tensor<1xf32>
    %661 = stablehlo.negate %660 : tensor<1xf32>
    %662 = stablehlo.slice %135 [0:1, 3:4] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %663 = stablehlo.reshape %662 : (tensor<1x1xf32>) -> tensor<1xf32>
    %664 = stablehlo.slice %135 [0:1, 3:4] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %665 = stablehlo.reshape %664 : (tensor<1x1xf32>) -> tensor<1xf32>
    %666 = stablehlo.add %663, %665 : tensor<1xf32>
    %667 = stablehlo.optimization_barrier %666 : tensor<1xf32>
    %cst_145 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %668 = stablehlo.broadcast_in_dim %cst_145, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %669 = stablehlo.multiply %668, %667 : tensor<1xf32>
    %670 = stablehlo.optimization_barrier %669 : tensor<1xf32>
    %671 = stablehlo.multiply %661, %670 : tensor<1xf32>
    %672 = stablehlo.optimization_barrier %671 : tensor<1xf32>
    %cst_146 = stablehlo.constant dense<1.000000e+01> : tensor<f32>
    %673 = stablehlo.broadcast_in_dim %cst_146, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %674 = stablehlo.power %673, %672 : tensor<1xf32>
    %675 = stablehlo.multiply %658, %674 : tensor<1xf32>
    %676 = stablehlo.optimization_barrier %675 : tensor<1xf32>
    %677 = stablehlo.add %648, %676 : tensor<1xf32>
    %678 = stablehlo.optimization_barrier %677 : tensor<1xf32>
    %679 = stablehlo.add %650, %658 : tensor<1xf32>
    %680 = stablehlo.optimization_barrier %679 : tensor<1xf32>
    %cst_147 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %681 = stablehlo.broadcast_in_dim %cst_147, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %682 = stablehlo.compare EQ, %680, %681, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %c_148 = stablehlo.constant dense<1> : tensor<i64>
    %683 = call @_where_1(%682, %c_148, %680) : (tensor<1xi1>, tensor<i64>, tensor<1xf32>) -> tensor<1xf32>
    %684 = stablehlo.divide %678, %683 : tensor<1xf32>
    %685 = stablehlo.optimization_barrier %684 : tensor<1xf32>
    %cst_149 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %686 = stablehlo.broadcast_in_dim %cst_149, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %687 = stablehlo.compare EQ, %558, %686, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %c_150 = stablehlo.constant dense<1> : tensor<i64>
    %688 = call @_where_1(%687, %c_150, %685) : (tensor<1xi1>, tensor<i64>, tensor<1xf32>) -> tensor<1xf32>
    %689 = stablehlo.slice %arg1 [0:1, 7:8] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %690 = stablehlo.reshape %689 : (tensor<1x1xf32>) -> tensor<1xf32>
    %cst_151 = stablehlo.constant dense<1.000000e+03> : tensor<f32>
    %691 = stablehlo.broadcast_in_dim %cst_151, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %692 = stablehlo.multiply %691, %690 : tensor<1xf32>
    %693 = stablehlo.optimization_barrier %692 : tensor<1xf32>
    %694 = stablehlo.slice %arg1 [0:1, 5:6] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %695 = stablehlo.reshape %694 : (tensor<1x1xf32>) -> tensor<1xf32>
    %696 = stablehlo.sine %695 : tensor<1xf32>
    %697 = stablehlo.multiply %693, %696 : tensor<1xf32>
    %698 = stablehlo.optimization_barrier %697 : tensor<1xf32>
    %699 = stablehlo.slice %arg1 [0:1, 3:4] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %700 = stablehlo.reshape %699 : (tensor<1x1xf32>) -> tensor<1xf32>
    %701 = stablehlo.multiply %700, %688 : tensor<1xf32>
    %702 = stablehlo.optimization_barrier %701 : tensor<1xf32>
    %703 = stablehlo.multiply %698, %702 : tensor<1xf32>
    %704 = stablehlo.optimization_barrier %703 : tensor<1xf32>
    %705 = stablehlo.multiply %704, %558 : tensor<1xf32>
    %706 = stablehlo.optimization_barrier %705 : tensor<1xf32>
    %707 = stablehlo.compare GE, %549, %556, SIGNED : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %c_152 = stablehlo.constant dense<0> : tensor<i64>
    %708 = call @_where_1(%707, %c_152, %706) : (tensor<1xi1>, tensor<i64>, tensor<1xf32>) -> tensor<1xf32>
    %709 = stablehlo.broadcast_in_dim %566, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %710 = stablehlo.broadcast_in_dim %598, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %711 = stablehlo.broadcast_in_dim %628, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %712 = stablehlo.broadcast_in_dim %658, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %713 = stablehlo.concatenate %709, %710, %711, %712, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<4x1xf32>
    %c_153 = stablehlo.constant dense<0> : tensor<i64>
    %714 = call @_where_4(%429, %708, %c_153) : (tensor<1xi1>, tensor<1xf32>, tensor<i64>) -> tensor<1xf32>
    %c_154 = stablehlo.constant dense<1> : tensor<i64>
    %715 = call @_where_4(%429, %688, %c_154) : (tensor<1xi1>, tensor<1xf32>, tensor<i64>) -> tensor<1xf32>
    %c_155 = stablehlo.constant dense<0> : tensor<i64>
    %716 = call @_where_5(%429, %713, %c_155) : (tensor<1xi1>, tensor<4x1xf32>, tensor<i64>) -> tensor<1x4xf32>
    %c_156 = stablehlo.constant dense<0> : tensor<i32>
    %717 = stablehlo.broadcast_in_dim %c_156, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %718 = stablehlo.compare GE, %300, %717, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %719 = stablehlo.convert %718 : (tensor<1xi1>) -> tensor<1xf32>
    %720 = stablehlo.convert %429 : (tensor<1xi1>) -> tensor<1xf32>
    %721 = stablehlo.convert %302 : (tensor<1xi32>) -> tensor<1xf32>
    %722 = stablehlo.convert %96 : (tensor<1xi64>) -> tensor<1xf32>
    %723 = stablehlo.slice %arg1 [0:1, 9:10] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %724 = stablehlo.reshape %723 : (tensor<1x1xf32>) -> tensor<1xf32>
    %725 = stablehlo.slice %arg1 [0:1, 0:1] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %726 = stablehlo.reshape %725 : (tensor<1x1xf32>) -> tensor<1xf32>
    %727 = stablehlo.negate %726 : tensor<1xf32>
    %728 = stablehlo.slice %arg1 [0:1, 1:2] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %729 = stablehlo.reshape %728 : (tensor<1x1xf32>) -> tensor<1xf32>
    %730 = stablehlo.divide %247, %729 : tensor<1xf32>
    %731 = stablehlo.optimization_barrier %730 : tensor<1xf32>
    %732 = stablehlo.log_plus_one %731 : tensor<1xf32>
    %733 = stablehlo.negate %732 : tensor<1xf32>
    %734 = stablehlo.slice %arg1 [0:1, 2:3] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %735 = stablehlo.reshape %734 : (tensor<1x1xf32>) -> tensor<1xf32>
    %736 = stablehlo.divide %733, %735 : tensor<1xf32>
    %737 = stablehlo.optimization_barrier %736 : tensor<1xf32>
    %738 = stablehlo.exponential_minus_one %737 : tensor<1xf32>
    %739 = stablehlo.multiply %727, %738 : tensor<1xf32>
    %740 = stablehlo.optimization_barrier %739 : tensor<1xf32>
    %741 = stablehlo.maximum %724, %740 : tensor<1xf32>
    %c_157 = stablehlo.constant dense<1> : tensor<i64>
    %742 = stablehlo.broadcast_in_dim %c_157, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %743 = stablehlo.add %549, %742 : tensor<1xi64>
    %c_158 = stablehlo.constant dense<-1> : tensor<i64>
    %744 = call @_where_6(%429, %743, %c_158) : (tensor<1xi1>, tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
    %745 = stablehlo.convert %744 : (tensor<1xi64>) -> tensor<1xf32>
    %746 = stablehlo.broadcast_in_dim %247, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %747 = stablehlo.broadcast_in_dim %315, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %748 = stablehlo.broadcast_in_dim %430, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %749 = stablehlo.broadcast_in_dim %719, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %750 = stablehlo.broadcast_in_dim %720, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %751 = stablehlo.broadcast_in_dim %721, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %752 = stablehlo.broadcast_in_dim %722, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %753 = stablehlo.broadcast_in_dim %537, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %754 = stablehlo.broadcast_in_dim %714, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %755 = stablehlo.broadcast_in_dim %526, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %756 = stablehlo.broadcast_in_dim %715, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %757 = stablehlo.broadcast_in_dim %741, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %758 = stablehlo.broadcast_in_dim %745, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %759 = stablehlo.concatenate %746, %747, %748, %749, %750, %751, %752, %753, %754, %755, %756, %757, %758, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<13x1xf32>
    %760 = stablehlo.transpose %759, dims = [1, 0] : (tensor<13x1xf32>) -> tensor<1x13xf32>
    %761 = stablehlo.transpose %542, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %762 = stablehlo.concatenate %760, %761, %716, %135, %118, dim = 1 : (tensor<1x13xf32>, tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x29xf32>
    return %762, %48 : tensor<1x29xf32>, tensor<1x4x9xf32>
  }
  func.func private @_where(%arg0: tensor<1xi1>, %arg1: tensor<i64>, %arg2: tensor<i32>) -> tensor<1xi32> {
    %0 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.select %arg0, %2, %1 : tensor<1xi1>, tensor<1xi32>
    return %3 : tensor<1xi32>
  }
  func.func private @_where_0(%arg0: tensor<1xi1>, %arg1: tensor<i64>, %arg2: tensor<1xi32>) -> tensor<1xi32> {
    %0 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<1xi1>, tensor<1xi32>
    return %2 : tensor<1xi32>
  }
  func.func private @clip(%arg0: tensor<1xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1xi32> {
    %0 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1xi32>
    %3 = stablehlo.convert %arg2 : (tensor<i64>) -> tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %5 = stablehlo.minimum %4, %2 : tensor<1xi32>
    return %5 : tensor<1xi32>
  }
  func.func private @_where_1(%arg0: tensor<1xi1>, %arg1: tensor<i64>, %arg2: tensor<1xf32>) -> tensor<1xf32> {
    %0 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<1xi1>, tensor<1xf32>
    return %2 : tensor<1xf32>
  }
  func.func private @clip_2(%arg0: tensor<1xf32>, %arg1: tensor<i64>, %arg2: tensor<1xf32>) -> tensor<1xf32> {
    %0 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1xf32>
    %3 = stablehlo.minimum %arg2, %2 : tensor<1xf32>
    return %3 : tensor<1xf32>
  }
  func.func private @_where_3(%arg0: tensor<1xi1>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1xi1>, tensor<1xf32>
    return %0 : tensor<1xf32>
  }
  func.func private @_where_4(%arg0: tensor<1xi1>, %arg1: tensor<1xf32>, %arg2: tensor<i64>) -> tensor<1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i64>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<1xi1>, tensor<1xf32>
    return %2 : tensor<1xf32>
  }
  func.func private @_where_5(%arg0: tensor<1xi1>, %arg1: tensor<4x1xf32>, %arg2: tensor<i64>) -> tensor<1x4xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i64>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
    %3 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1xi1>) -> tensor<1x4xi1>
    %5 = stablehlo.select %4, %3, %2 : tensor<1x4xi1>, tensor<1x4xf32>
    return %5 : tensor<1x4xf32>
  }
  func.func private @_where_6(%arg0: tensor<1xi1>, %arg1: tensor<1xi64>, %arg2: tensor<i64>) -> tensor<1xi64> {
    %0 = stablehlo.convert %arg2 : tensor<i64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<1xi1>, tensor<1xi64>
    return %2 : tensor<1xi64>
  }
}
