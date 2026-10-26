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
    %39 = stablehlo.slice %arg0 [0:1, 0:1, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %40 = stablehlo.reshape %39 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %41 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %42 = stablehlo.add %41, %40 : tensor<1xf32>
    %43 = stablehlo.optimization_barrier %42 : tensor<1xf32>
    %44 = stablehlo.slice %arg0 [0:1, 1:2, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %45 = stablehlo.reshape %44 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %46 = stablehlo.add %43, %45 : tensor<1xf32>
    %47 = stablehlo.optimization_barrier %46 : tensor<1xf32>
    %48 = stablehlo.slice %arg0 [0:1, 2:3, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %49 = stablehlo.reshape %48 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %50 = stablehlo.add %47, %49 : tensor<1xf32>
    %51 = stablehlo.optimization_barrier %50 : tensor<1xf32>
    %52 = stablehlo.slice %arg0 [0:1, 3:4, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %53 = stablehlo.reshape %52 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %54 = stablehlo.add %51, %53 : tensor<1xf32>
    %55 = stablehlo.optimization_barrier %54 : tensor<1xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %56 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %58 = stablehlo.broadcast_in_dim %43, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %59 = stablehlo.broadcast_in_dim %47, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %60 = stablehlo.broadcast_in_dim %51, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %61 = stablehlo.broadcast_in_dim %55, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %62 = stablehlo.concatenate %57, %58, %59, %60, %61, dim = 1 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x5xf32>
    %63 = stablehlo.slice %62 [0:1, 0:4] : (tensor<1x5xf32>) -> tensor<1x4xf32>
    %64 = stablehlo.slice %arg0 [0:1, 0:4, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %65 = stablehlo.reshape %64 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %cst_5 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %66 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %67 = stablehlo.multiply %66, %65 : tensor<1x4xf32>
    %68 = stablehlo.optimization_barrier %67 : tensor<1x4xf32>
    %69 = stablehlo.add %63, %68 : tensor<1x4xf32>
    %70 = stablehlo.optimization_barrier %69 : tensor<1x4xf32>
    %71 = stablehlo.slice %62 [0:1, 1:5] : (tensor<1x5xf32>) -> tensor<1x4xf32>
    %72 = stablehlo.slice %arg1 [0:1, 4:5] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %73 = stablehlo.reshape %72 : (tensor<1x1xf32>) -> tensor<1xf32>
    %74 = stablehlo.slice %62 [0:1, 4:5] : (tensor<1x5xf32>) -> tensor<1x1xf32>
    %75 = stablehlo.reshape %74 : (tensor<1x1xf32>) -> tensor<1xf32>
    %cst_6 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %76 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %77 = stablehlo.maximum %75, %76 : tensor<1xf32>
    %cst_7 = stablehlo.constant dense<1.90734863E-6> : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %79 = stablehlo.multiply %78, %77 : tensor<1xf32>
    %80 = stablehlo.add %73, %79 : tensor<1xf32>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %82 = stablehlo.broadcast_in_dim %81, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
    %83 = stablehlo.compare LE, %71, %82, FLOAT : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xi1>
    %84 = stablehlo.convert %83 : (tensor<1x4xi1>) -> tensor<1x4xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %85 = stablehlo.reduce(%84 init: %c) applies stablehlo.add across dimensions = [1] : (tensor<1x4xi32>, tensor<i32>) -> tensor<1xi32>
    %86 = stablehlo.slice %arg0 [0:1, 0:4, 0:1] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %87 = stablehlo.reshape %86 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+03> : tensor<f32>
    %88 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %89 = stablehlo.divide %87, %88 : tensor<1x4xf32>
    %90 = stablehlo.optimization_barrier %89 : tensor<1x4xf32>
    %91 = stablehlo.slice %arg0 [0:1, 0:4, 1:2] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %92 = stablehlo.reshape %91 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %cst_9 = stablehlo.constant dense<9.170000e+02> : tensor<f32>
    %93 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %94 = stablehlo.divide %92, %93 : tensor<1x4xf32>
    %95 = stablehlo.optimization_barrier %94 : tensor<1x4xf32>
    %96 = stablehlo.add %90, %95 : tensor<1x4xf32>
    %97 = stablehlo.optimization_barrier %96 : tensor<1x4xf32>
    %98 = stablehlo.slice %arg0 [0:1, 0:4, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %99 = stablehlo.reshape %98 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %100 = stablehlo.slice %arg1 [0:1, 0:1] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %101 = stablehlo.reshape %100 : (tensor<1x1xf32>) -> tensor<1xf32>
    %102 = stablehlo.broadcast_in_dim %101, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %103 = stablehlo.broadcast_in_dim %102, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
    %104 = stablehlo.multiply %99, %103 : tensor<1x4xf32>
    %105 = stablehlo.optimization_barrier %104 : tensor<1x4xf32>
    %106 = stablehlo.divide %97, %105 : tensor<1x4xf32>
    %107 = stablehlo.optimization_barrier %106 : tensor<1x4xf32>
    %108 = stablehlo.slice %arg0 [0:1, 0:4, 1:2] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %109 = stablehlo.reshape %108 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %110 = stablehlo.slice %arg0 [0:1, 0:4, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x4x1xf32>
    %111 = stablehlo.reshape %110 : (tensor<1x4x1xf32>) -> tensor<1x4xf32>
    %cst_10 = stablehlo.constant dense<9.170000e+02> : tensor<f32>
    %112 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %113 = stablehlo.multiply %112, %111 : tensor<1x4xf32>
    %114 = stablehlo.optimization_barrier %113 : tensor<1x4xf32>
    %115 = stablehlo.slice %arg1 [0:1, 0:1] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %116 = stablehlo.reshape %115 : (tensor<1x1xf32>) -> tensor<1xf32>
    %117 = stablehlo.broadcast_in_dim %116, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %118 = stablehlo.broadcast_in_dim %117, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
    %119 = stablehlo.multiply %114, %118 : tensor<1x4xf32>
    %120 = stablehlo.optimization_barrier %119 : tensor<1x4xf32>
    %121 = stablehlo.divide %109, %120 : tensor<1x4xf32>
    %122 = stablehlo.optimization_barrier %121 : tensor<1x4xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %123 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x4xf32>
    %124 = stablehlo.minimum %123, %122 : tensor<1x4xf32>
    %125 = stablehlo.slice %arg1 [0:1, 4:5] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %126 = stablehlo.reshape %125 : (tensor<1x1xf32>) -> tensor<1xf32>
    %127 = stablehlo.slice %arg1 [0:1, 8:9] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %128 = stablehlo.reshape %127 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_12 = stablehlo.constant dense<0> : tensor<i32>
    %129 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %130 = stablehlo.compare GT, %85, %129, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %131 = stablehlo.slice %107 [0:1, 0:1] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %132 = stablehlo.reshape %131 : (tensor<1x1xf32>) -> tensor<1xf32>
    %133 = stablehlo.compare LE, %132, %128, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %134 = stablehlo.and %130, %133 : tensor<1xi1>
    %c_13 = stablehlo.constant dense<0> : tensor<i32>
    %c_14 = stablehlo.constant dense<-1> : tensor<i32>
    %135 = call @_where(%134, %c_13, %c_14) : (tensor<1xi1>, tensor<i32>, tensor<i32>) -> tensor<1xi32>
    %c_15 = stablehlo.constant dense<1> : tensor<i32>
    %136 = stablehlo.broadcast_in_dim %c_15, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %137 = stablehlo.compare GT, %85, %136, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %138 = stablehlo.slice %107 [0:1, 1:2] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %139 = stablehlo.reshape %138 : (tensor<1x1xf32>) -> tensor<1xf32>
    %140 = stablehlo.compare LE, %139, %128, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %141 = stablehlo.and %137, %140 : tensor<1xi1>
    %c_16 = stablehlo.constant dense<1> : tensor<i32>
    %142 = call @_where_0(%141, %c_16, %135) : (tensor<1xi1>, tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
    %c_17 = stablehlo.constant dense<2> : tensor<i32>
    %143 = stablehlo.broadcast_in_dim %c_17, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %144 = stablehlo.compare GT, %85, %143, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %145 = stablehlo.slice %107 [0:1, 2:3] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %146 = stablehlo.reshape %145 : (tensor<1x1xf32>) -> tensor<1xf32>
    %147 = stablehlo.compare LE, %146, %128, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %148 = stablehlo.and %144, %147 : tensor<1xi1>
    %c_18 = stablehlo.constant dense<2> : tensor<i32>
    %149 = call @_where_0(%148, %c_18, %142) : (tensor<1xi1>, tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
    %c_19 = stablehlo.constant dense<3> : tensor<i32>
    %150 = stablehlo.broadcast_in_dim %c_19, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %151 = stablehlo.compare GT, %85, %150, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %152 = stablehlo.slice %107 [0:1, 3:4] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %153 = stablehlo.reshape %152 : (tensor<1x1xf32>) -> tensor<1xf32>
    %154 = stablehlo.compare LE, %153, %128, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %155 = stablehlo.and %151, %154 : tensor<1xi1>
    %c_20 = stablehlo.constant dense<3> : tensor<i32>
    %156 = call @_where_0(%155, %c_20, %149) : (tensor<1xi1>, tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
    %c_21 = stablehlo.constant dense<0> : tensor<i32>
    %c_22 = stablehlo.constant dense<2> : tensor<i32>
    %157 = call @clip(%156, %c_21, %c_22) : (tensor<1xi32>, tensor<i32>, tensor<i32>) -> tensor<1xi32>
    %c_23 = stablehlo.constant dense<1> : tensor<i32>
    %158 = stablehlo.broadcast_in_dim %c_23, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %159 = stablehlo.add %157, %158 : tensor<1xi32>
    %c_24 = stablehlo.constant dense<0> : tensor<i32>
    %160 = stablehlo.broadcast_in_dim %c_24, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %161 = stablehlo.compare LT, %159, %160, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_25 = stablehlo.constant dense<4> : tensor<i32>
    %162 = stablehlo.broadcast_in_dim %c_25, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %163 = stablehlo.add %159, %162 : tensor<1xi32>
    %164 = stablehlo.select %161, %163, %159 : tensor<1xi1>, tensor<1xi32>
    %165 = stablehlo.broadcast_in_dim %164, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %166 = "stablehlo.gather"(%107, %165) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %167 = stablehlo.reshape %166 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_26 = stablehlo.constant dense<0> : tensor<i32>
    %168 = stablehlo.broadcast_in_dim %c_26, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %169 = stablehlo.compare LT, %157, %168, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_27 = stablehlo.constant dense<4> : tensor<i32>
    %170 = stablehlo.broadcast_in_dim %c_27, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %171 = stablehlo.add %157, %170 : tensor<1xi32>
    %172 = stablehlo.select %169, %171, %157 : tensor<1xi1>, tensor<1xi32>
    %173 = stablehlo.broadcast_in_dim %172, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %174 = "stablehlo.gather"(%107, %173) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %175 = stablehlo.reshape %174 : (tensor<1x1xf32>) -> tensor<1xf32>
    %176 = stablehlo.subtract %167, %175 : tensor<1xf32>
    %177 = stablehlo.optimization_barrier %176 : tensor<1xf32>
    %c_28 = stablehlo.constant dense<0> : tensor<i32>
    %178 = stablehlo.broadcast_in_dim %c_28, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %179 = stablehlo.compare LT, %157, %178, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_29 = stablehlo.constant dense<4> : tensor<i32>
    %180 = stablehlo.broadcast_in_dim %c_29, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %181 = stablehlo.add %157, %180 : tensor<1xi32>
    %182 = stablehlo.select %179, %181, %157 : tensor<1xi1>, tensor<1xi32>
    %183 = stablehlo.broadcast_in_dim %182, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %184 = "stablehlo.gather"(%107, %183) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %185 = stablehlo.reshape %184 : (tensor<1x1xf32>) -> tensor<1xf32>
    %186 = stablehlo.subtract %128, %185 : tensor<1xf32>
    %187 = stablehlo.optimization_barrier %186 : tensor<1xf32>
    %cst_30 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %188 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %189 = stablehlo.compare EQ, %177, %188, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %c_31 = stablehlo.constant dense<1> : tensor<i32>
    %190 = call @_where_1(%189, %c_31, %177) : (tensor<1xi1>, tensor<i32>, tensor<1xf32>) -> tensor<1xf32>
    %191 = stablehlo.divide %187, %190 : tensor<1xf32>
    %192 = stablehlo.optimization_barrier %191 : tensor<1xf32>
    %c_32 = stablehlo.constant dense<0> : tensor<i32>
    %193 = stablehlo.broadcast_in_dim %c_32, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %194 = stablehlo.compare LT, %157, %193, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_33 = stablehlo.constant dense<4> : tensor<i32>
    %195 = stablehlo.broadcast_in_dim %c_33, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %196 = stablehlo.add %157, %195 : tensor<1xi32>
    %197 = stablehlo.select %194, %196, %157 : tensor<1xi1>, tensor<1xi32>
    %198 = stablehlo.broadcast_in_dim %197, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %199 = "stablehlo.gather"(%70, %198) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %200 = stablehlo.reshape %199 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_34 = stablehlo.constant dense<1> : tensor<i32>
    %201 = stablehlo.broadcast_in_dim %c_34, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %202 = stablehlo.add %157, %201 : tensor<1xi32>
    %c_35 = stablehlo.constant dense<0> : tensor<i32>
    %203 = stablehlo.broadcast_in_dim %c_35, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %204 = stablehlo.compare LT, %202, %203, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_36 = stablehlo.constant dense<4> : tensor<i32>
    %205 = stablehlo.broadcast_in_dim %c_36, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %206 = stablehlo.add %202, %205 : tensor<1xi32>
    %207 = stablehlo.select %204, %206, %202 : tensor<1xi1>, tensor<1xi32>
    %208 = stablehlo.broadcast_in_dim %207, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %209 = "stablehlo.gather"(%70, %208) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %210 = stablehlo.reshape %209 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_37 = stablehlo.constant dense<0> : tensor<i32>
    %211 = stablehlo.broadcast_in_dim %c_37, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %212 = stablehlo.compare LT, %157, %211, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_38 = stablehlo.constant dense<4> : tensor<i32>
    %213 = stablehlo.broadcast_in_dim %c_38, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %214 = stablehlo.add %157, %213 : tensor<1xi32>
    %215 = stablehlo.select %212, %214, %157 : tensor<1xi1>, tensor<1xi32>
    %216 = stablehlo.broadcast_in_dim %215, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %217 = "stablehlo.gather"(%70, %216) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %218 = stablehlo.reshape %217 : (tensor<1x1xf32>) -> tensor<1xf32>
    %219 = stablehlo.subtract %210, %218 : tensor<1xf32>
    %220 = stablehlo.optimization_barrier %219 : tensor<1xf32>
    %221 = stablehlo.multiply %192, %220 : tensor<1xf32>
    %222 = stablehlo.optimization_barrier %221 : tensor<1xf32>
    %223 = stablehlo.add %200, %222 : tensor<1xf32>
    %224 = stablehlo.optimization_barrier %223 : tensor<1xf32>
    %c_39 = stablehlo.constant dense<0> : tensor<i32>
    %225 = call @clip_2(%224, %c_39, %126) : (tensor<1xf32>, tensor<i32>, tensor<1xf32>) -> tensor<1xf32>
    %c_40 = stablehlo.constant dense<0> : tensor<i32>
    %226 = stablehlo.broadcast_in_dim %c_40, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %227 = stablehlo.compare LE, %156, %226, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %228 = stablehlo.slice %62 [0:1, 1:2] : (tensor<1x5xf32>) -> tensor<1x1xf32>
    %229 = stablehlo.reshape %228 : (tensor<1x1xf32>) -> tensor<1xf32>
    %230 = stablehlo.minimum %229, %126 : tensor<1xf32>
    %c_41 = stablehlo.constant dense<1> : tensor<i32>
    %231 = stablehlo.broadcast_in_dim %c_41, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %232 = stablehlo.subtract %85, %231 : tensor<1xi32>
    %233 = stablehlo.compare EQ, %156, %232, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %234 = call @_where_3(%233, %126, %225) : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %235 = call @_where_3(%227, %230, %234) : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %c_42 = stablehlo.constant dense<1> : tensor<i32>
    %236 = stablehlo.broadcast_in_dim %c_42, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %237 = stablehlo.compare GT, %85, %236, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %238 = stablehlo.slice %arg0 [0:1, 0:1, 3:4] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %239 = stablehlo.reshape %238 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %cst_43 = stablehlo.constant dense<2.731500e+02> : tensor<f32>
    %240 = stablehlo.broadcast_in_dim %cst_43, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %241 = stablehlo.compare GT, %239, %240, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %242 = stablehlo.and %237, %241 : tensor<1xi1>
    %243 = stablehlo.slice %arg0 [0:1, 1:2, 3:4] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %244 = stablehlo.reshape %243 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %cst_44 = stablehlo.constant dense<2.731500e+02> : tensor<f32>
    %245 = stablehlo.broadcast_in_dim %cst_44, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %246 = stablehlo.compare LE, %244, %245, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %247 = stablehlo.and %242, %246 : tensor<1xi1>
    %c_45 = stablehlo.constant dense<-1> : tensor<i32>
    %c_46 = stablehlo.constant dense<0> : tensor<i32>
    %248 = stablehlo.compare LT, %c_45, %c_46, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %249 = stablehlo.broadcast_in_dim %248, dims = [] : (tensor<i1>) -> tensor<1xi1>
    %250 = stablehlo.and %247, %249 : tensor<1xi1>
    %c_47 = stablehlo.constant dense<1> : tensor<i32>
    %c_48 = stablehlo.constant dense<-1> : tensor<i32>
    %251 = call @_where(%250, %c_47, %c_48) : (tensor<1xi1>, tensor<i32>, tensor<i32>) -> tensor<1xi32>
    %252 = stablehlo.convert %247 : (tensor<1xi1>) -> tensor<1xi32>
    %c_49 = stablehlo.constant dense<0> : tensor<i32>
    %253 = stablehlo.broadcast_in_dim %c_49, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %254 = stablehlo.add %253, %252 : tensor<1xi32>
    %c_50 = stablehlo.constant dense<2> : tensor<i32>
    %255 = stablehlo.broadcast_in_dim %c_50, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %256 = stablehlo.compare GT, %85, %255, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %257 = stablehlo.slice %arg0 [0:1, 1:2, 3:4] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %258 = stablehlo.reshape %257 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %cst_51 = stablehlo.constant dense<2.731500e+02> : tensor<f32>
    %259 = stablehlo.broadcast_in_dim %cst_51, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %260 = stablehlo.compare GT, %258, %259, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %261 = stablehlo.and %256, %260 : tensor<1xi1>
    %262 = stablehlo.slice %arg0 [0:1, 2:3, 3:4] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %263 = stablehlo.reshape %262 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %cst_52 = stablehlo.constant dense<2.731500e+02> : tensor<f32>
    %264 = stablehlo.broadcast_in_dim %cst_52, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %265 = stablehlo.compare LE, %263, %264, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %266 = stablehlo.and %261, %265 : tensor<1xi1>
    %c_53 = stablehlo.constant dense<0> : tensor<i32>
    %267 = stablehlo.broadcast_in_dim %c_53, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %268 = stablehlo.compare LT, %251, %267, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %269 = stablehlo.and %266, %268 : tensor<1xi1>
    %c_54 = stablehlo.constant dense<2> : tensor<i32>
    %270 = call @_where_0(%269, %c_54, %251) : (tensor<1xi1>, tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
    %271 = stablehlo.convert %266 : (tensor<1xi1>) -> tensor<1xi32>
    %272 = stablehlo.add %254, %271 : tensor<1xi32>
    %c_55 = stablehlo.constant dense<3> : tensor<i32>
    %273 = stablehlo.broadcast_in_dim %c_55, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %274 = stablehlo.compare GT, %85, %273, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %275 = stablehlo.slice %arg0 [0:1, 2:3, 3:4] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %276 = stablehlo.reshape %275 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %cst_56 = stablehlo.constant dense<2.731500e+02> : tensor<f32>
    %277 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %278 = stablehlo.compare GT, %276, %277, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %279 = stablehlo.and %274, %278 : tensor<1xi1>
    %280 = stablehlo.slice %arg0 [0:1, 3:4, 3:4] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %281 = stablehlo.reshape %280 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %cst_57 = stablehlo.constant dense<2.731500e+02> : tensor<f32>
    %282 = stablehlo.broadcast_in_dim %cst_57, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %283 = stablehlo.compare LE, %281, %282, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %284 = stablehlo.and %279, %283 : tensor<1xi1>
    %c_58 = stablehlo.constant dense<0> : tensor<i32>
    %285 = stablehlo.broadcast_in_dim %c_58, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %286 = stablehlo.compare LT, %270, %285, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %287 = stablehlo.and %284, %286 : tensor<1xi1>
    %c_59 = stablehlo.constant dense<3> : tensor<i32>
    %288 = call @_where_0(%287, %c_59, %270) : (tensor<1xi1>, tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
    %289 = stablehlo.convert %284 : (tensor<1xi1>) -> tensor<1xi32>
    %290 = stablehlo.add %272, %289 : tensor<1xi32>
    %c_60 = stablehlo.constant dense<0> : tensor<i32>
    %291 = stablehlo.broadcast_in_dim %c_60, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %292 = stablehlo.compare GE, %288, %291, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_61 = stablehlo.constant dense<0> : tensor<i32>
    %293 = stablehlo.broadcast_in_dim %c_61, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %294 = stablehlo.maximum %288, %293 : tensor<1xi32>
    %c_62 = stablehlo.constant dense<0> : tensor<i32>
    %295 = stablehlo.broadcast_in_dim %c_62, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %296 = stablehlo.compare LT, %294, %295, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_63 = stablehlo.constant dense<4> : tensor<i32>
    %297 = stablehlo.broadcast_in_dim %c_63, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %298 = stablehlo.add %294, %297 : tensor<1xi32>
    %299 = stablehlo.select %296, %298, %294 : tensor<1xi1>, tensor<1xi32>
    %300 = stablehlo.broadcast_in_dim %299, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %301 = "stablehlo.gather"(%70, %300) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %302 = stablehlo.reshape %301 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_64 = stablehlo.constant dense<0> : tensor<i32>
    %303 = call @_where_4(%292, %302, %c_64) : (tensor<1xi1>, tensor<1xf32>, tensor<i32>) -> tensor<1xf32>
    %c_65 = stablehlo.constant dense<1> : tensor<i32>
    %304 = stablehlo.broadcast_in_dim %c_65, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %305 = stablehlo.add %288, %304 : tensor<1xi32>
    %306 = stablehlo.slice %arg1 [0:1, 8:9] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %307 = stablehlo.reshape %306 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_66 = stablehlo.constant dense<0> : tensor<i32>
    %308 = stablehlo.broadcast_in_dim %c_66, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %309 = stablehlo.compare GT, %305, %308, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %310 = stablehlo.slice %107 [0:1, 0:1] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %311 = stablehlo.reshape %310 : (tensor<1x1xf32>) -> tensor<1xf32>
    %312 = stablehlo.compare LE, %311, %307, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %313 = stablehlo.and %309, %312 : tensor<1xi1>
    %c_67 = stablehlo.constant dense<0> : tensor<i32>
    %c_68 = stablehlo.constant dense<-1> : tensor<i32>
    %314 = call @_where(%313, %c_67, %c_68) : (tensor<1xi1>, tensor<i32>, tensor<i32>) -> tensor<1xi32>
    %c_69 = stablehlo.constant dense<1> : tensor<i32>
    %315 = stablehlo.broadcast_in_dim %c_69, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %316 = stablehlo.compare GT, %305, %315, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %317 = stablehlo.slice %107 [0:1, 1:2] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %318 = stablehlo.reshape %317 : (tensor<1x1xf32>) -> tensor<1xf32>
    %319 = stablehlo.compare LE, %318, %307, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %320 = stablehlo.and %316, %319 : tensor<1xi1>
    %c_70 = stablehlo.constant dense<1> : tensor<i32>
    %321 = call @_where_0(%320, %c_70, %314) : (tensor<1xi1>, tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
    %c_71 = stablehlo.constant dense<2> : tensor<i32>
    %322 = stablehlo.broadcast_in_dim %c_71, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %323 = stablehlo.compare GT, %305, %322, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %324 = stablehlo.slice %107 [0:1, 2:3] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %325 = stablehlo.reshape %324 : (tensor<1x1xf32>) -> tensor<1xf32>
    %326 = stablehlo.compare LE, %325, %307, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %327 = stablehlo.and %323, %326 : tensor<1xi1>
    %c_72 = stablehlo.constant dense<2> : tensor<i32>
    %328 = call @_where_0(%327, %c_72, %321) : (tensor<1xi1>, tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
    %c_73 = stablehlo.constant dense<3> : tensor<i32>
    %329 = stablehlo.broadcast_in_dim %c_73, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %330 = stablehlo.compare GT, %305, %329, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %331 = stablehlo.slice %107 [0:1, 3:4] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %332 = stablehlo.reshape %331 : (tensor<1x1xf32>) -> tensor<1xf32>
    %333 = stablehlo.compare LE, %332, %307, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %334 = stablehlo.and %330, %333 : tensor<1xi1>
    %c_74 = stablehlo.constant dense<3> : tensor<i32>
    %335 = call @_where_0(%334, %c_74, %328) : (tensor<1xi1>, tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
    %c_75 = stablehlo.constant dense<0> : tensor<i32>
    %c_76 = stablehlo.constant dense<2> : tensor<i32>
    %336 = call @clip(%335, %c_75, %c_76) : (tensor<1xi32>, tensor<i32>, tensor<i32>) -> tensor<1xi32>
    %c_77 = stablehlo.constant dense<1> : tensor<i32>
    %337 = stablehlo.broadcast_in_dim %c_77, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %338 = stablehlo.add %336, %337 : tensor<1xi32>
    %c_78 = stablehlo.constant dense<0> : tensor<i32>
    %339 = stablehlo.broadcast_in_dim %c_78, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %340 = stablehlo.compare LT, %338, %339, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_79 = stablehlo.constant dense<4> : tensor<i32>
    %341 = stablehlo.broadcast_in_dim %c_79, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %342 = stablehlo.add %338, %341 : tensor<1xi32>
    %343 = stablehlo.select %340, %342, %338 : tensor<1xi1>, tensor<1xi32>
    %344 = stablehlo.broadcast_in_dim %343, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %345 = "stablehlo.gather"(%107, %344) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %346 = stablehlo.reshape %345 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_80 = stablehlo.constant dense<0> : tensor<i32>
    %347 = stablehlo.broadcast_in_dim %c_80, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %348 = stablehlo.compare LT, %336, %347, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_81 = stablehlo.constant dense<4> : tensor<i32>
    %349 = stablehlo.broadcast_in_dim %c_81, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %350 = stablehlo.add %336, %349 : tensor<1xi32>
    %351 = stablehlo.select %348, %350, %336 : tensor<1xi1>, tensor<1xi32>
    %352 = stablehlo.broadcast_in_dim %351, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %353 = "stablehlo.gather"(%107, %352) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %354 = stablehlo.reshape %353 : (tensor<1x1xf32>) -> tensor<1xf32>
    %355 = stablehlo.subtract %346, %354 : tensor<1xf32>
    %356 = stablehlo.optimization_barrier %355 : tensor<1xf32>
    %c_82 = stablehlo.constant dense<0> : tensor<i32>
    %357 = stablehlo.broadcast_in_dim %c_82, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %358 = stablehlo.compare LT, %336, %357, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_83 = stablehlo.constant dense<4> : tensor<i32>
    %359 = stablehlo.broadcast_in_dim %c_83, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %360 = stablehlo.add %336, %359 : tensor<1xi32>
    %361 = stablehlo.select %358, %360, %336 : tensor<1xi1>, tensor<1xi32>
    %362 = stablehlo.broadcast_in_dim %361, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %363 = "stablehlo.gather"(%107, %362) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %364 = stablehlo.reshape %363 : (tensor<1x1xf32>) -> tensor<1xf32>
    %365 = stablehlo.subtract %307, %364 : tensor<1xf32>
    %366 = stablehlo.optimization_barrier %365 : tensor<1xf32>
    %cst_84 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %367 = stablehlo.broadcast_in_dim %cst_84, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %368 = stablehlo.compare EQ, %356, %367, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %c_85 = stablehlo.constant dense<1> : tensor<i32>
    %369 = call @_where_1(%368, %c_85, %356) : (tensor<1xi1>, tensor<i32>, tensor<1xf32>) -> tensor<1xf32>
    %370 = stablehlo.divide %366, %369 : tensor<1xf32>
    %371 = stablehlo.optimization_barrier %370 : tensor<1xf32>
    %c_86 = stablehlo.constant dense<0> : tensor<i32>
    %372 = stablehlo.broadcast_in_dim %c_86, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %373 = stablehlo.compare LT, %336, %372, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_87 = stablehlo.constant dense<4> : tensor<i32>
    %374 = stablehlo.broadcast_in_dim %c_87, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %375 = stablehlo.add %336, %374 : tensor<1xi32>
    %376 = stablehlo.select %373, %375, %336 : tensor<1xi1>, tensor<1xi32>
    %377 = stablehlo.broadcast_in_dim %376, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %378 = "stablehlo.gather"(%70, %377) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %379 = stablehlo.reshape %378 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_88 = stablehlo.constant dense<1> : tensor<i32>
    %380 = stablehlo.broadcast_in_dim %c_88, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %381 = stablehlo.add %336, %380 : tensor<1xi32>
    %c_89 = stablehlo.constant dense<0> : tensor<i32>
    %382 = stablehlo.broadcast_in_dim %c_89, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %383 = stablehlo.compare LT, %381, %382, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_90 = stablehlo.constant dense<4> : tensor<i32>
    %384 = stablehlo.broadcast_in_dim %c_90, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %385 = stablehlo.add %381, %384 : tensor<1xi32>
    %386 = stablehlo.select %383, %385, %381 : tensor<1xi1>, tensor<1xi32>
    %387 = stablehlo.broadcast_in_dim %386, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %388 = "stablehlo.gather"(%70, %387) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %389 = stablehlo.reshape %388 : (tensor<1x1xf32>) -> tensor<1xf32>
    %c_91 = stablehlo.constant dense<0> : tensor<i32>
    %390 = stablehlo.broadcast_in_dim %c_91, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %391 = stablehlo.compare LT, %336, %390, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_92 = stablehlo.constant dense<4> : tensor<i32>
    %392 = stablehlo.broadcast_in_dim %c_92, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %393 = stablehlo.add %336, %392 : tensor<1xi32>
    %394 = stablehlo.select %391, %393, %336 : tensor<1xi1>, tensor<1xi32>
    %395 = stablehlo.broadcast_in_dim %394, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %396 = "stablehlo.gather"(%70, %395) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<1x4xf32>, tensor<1x1xi32>) -> tensor<1x1xf32>
    %397 = stablehlo.reshape %396 : (tensor<1x1xf32>) -> tensor<1xf32>
    %398 = stablehlo.subtract %389, %397 : tensor<1xf32>
    %399 = stablehlo.optimization_barrier %398 : tensor<1xf32>
    %400 = stablehlo.multiply %371, %399 : tensor<1xf32>
    %401 = stablehlo.optimization_barrier %400 : tensor<1xf32>
    %402 = stablehlo.add %379, %401 : tensor<1xf32>
    %403 = stablehlo.optimization_barrier %402 : tensor<1xf32>
    %c_93 = stablehlo.constant dense<0> : tensor<i32>
    %404 = call @clip_2(%403, %c_93, %303) : (tensor<1xf32>, tensor<i32>, tensor<1xf32>) -> tensor<1xf32>
    %c_94 = stablehlo.constant dense<0> : tensor<i32>
    %405 = stablehlo.broadcast_in_dim %c_94, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %406 = stablehlo.compare LT, %335, %405, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_95 = stablehlo.constant dense<1> : tensor<i32>
    %407 = stablehlo.broadcast_in_dim %c_95, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %408 = stablehlo.subtract %305, %407 : tensor<1xi32>
    %409 = stablehlo.compare EQ, %335, %408, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %410 = call @_where_3(%409, %303, %404) : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %c_96 = stablehlo.constant dense<0> : tensor<i32>
    %411 = call @_where_1(%406, %c_96, %410) : (tensor<1xi1>, tensor<i32>, tensor<1xf32>) -> tensor<1xf32>
    %c_97 = stablehlo.constant dense<0> : tensor<i32>
    %412 = stablehlo.broadcast_in_dim %c_97, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %413 = stablehlo.compare GE, %288, %412, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %414 = stablehlo.compare GE, %235, %303, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %415 = stablehlo.and %413, %414 : tensor<1xi1>
    %416 = stablehlo.compare LT, %411, %303, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %417 = stablehlo.and %415, %416 : tensor<1xi1>
    %c_98 = stablehlo.constant dense<0> : tensor<i32>
    %418 = call @_where_4(%417, %411, %c_98) : (tensor<1xi1>, tensor<1xf32>, tensor<i32>) -> tensor<1xf32>
    %419 = stablehlo.slice %arg1 [0:1, 4:5] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %420 = stablehlo.reshape %419 : (tensor<1x1xf32>) -> tensor<1xf32>
    %421 = stablehlo.slice %62 [0:1, 1:5] : (tensor<1x5xf32>) -> tensor<1x4xf32>
    %422 = stablehlo.broadcast_in_dim %235, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %423 = stablehlo.broadcast_in_dim %422, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
    %424 = stablehlo.compare LT, %421, %423, FLOAT : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xi1>
    %425 = stablehlo.convert %424 : (tensor<1x4xi1>) -> tensor<1x4xi32>
    %c_99 = stablehlo.constant dense<0> : tensor<i32>
    %426 = stablehlo.reduce(%425 init: %c_99) applies stablehlo.add across dimensions = [1] : (tensor<1x4xi32>, tensor<i32>) -> tensor<1xi32>
    %427 = stablehlo.subtract %420, %235 : tensor<1xf32>
    %428 = stablehlo.optimization_barrier %427 : tensor<1xf32>
    %c_100 = stablehlo.constant dense<1> : tensor<i32>
    %429 = stablehlo.broadcast_in_dim %c_100, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %430 = stablehlo.subtract %426, %429 : tensor<1xi32>
    %c_101 = stablehlo.constant dense<0> : tensor<i32>
    %431 = stablehlo.broadcast_in_dim %c_101, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %432 = stablehlo.maximum %430, %431 : tensor<1xi32>
    %c_102 = stablehlo.constant dense<0> : tensor<i32>
    %433 = stablehlo.broadcast_in_dim %c_102, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %434 = stablehlo.compare LE, %432, %433, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %435 = stablehlo.slice %arg0 [0:1, 0:1, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %436 = stablehlo.reshape %435 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_103 = stablehlo.constant dense<0> : tensor<i32>
    %437 = call @_where_4(%434, %436, %c_103) : (tensor<1xi1>, tensor<1xf32>, tensor<i32>) -> tensor<1xf32>
    %438 = stablehlo.slice %124 [0:1, 0:1] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %439 = stablehlo.reshape %438 : (tensor<1x1xf32>) -> tensor<1xf32>
    %440 = stablehlo.multiply %437, %439 : tensor<1xf32>
    %441 = stablehlo.optimization_barrier %440 : tensor<1xf32>
    %cst_104 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %442 = stablehlo.broadcast_in_dim %cst_104, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %443 = stablehlo.add %442, %441 : tensor<1xf32>
    %444 = stablehlo.optimization_barrier %443 : tensor<1xf32>
    %cst_105 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %445 = stablehlo.broadcast_in_dim %cst_105, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %446 = stablehlo.add %445, %437 : tensor<1xf32>
    %447 = stablehlo.optimization_barrier %446 : tensor<1xf32>
    %c_106 = stablehlo.constant dense<1> : tensor<i32>
    %448 = stablehlo.broadcast_in_dim %c_106, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %449 = stablehlo.subtract %426, %448 : tensor<1xi32>
    %c_107 = stablehlo.constant dense<0> : tensor<i32>
    %450 = stablehlo.broadcast_in_dim %c_107, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %451 = stablehlo.maximum %449, %450 : tensor<1xi32>
    %c_108 = stablehlo.constant dense<1> : tensor<i32>
    %452 = stablehlo.broadcast_in_dim %c_108, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %453 = stablehlo.compare LE, %451, %452, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %454 = stablehlo.slice %arg0 [0:1, 1:2, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %455 = stablehlo.reshape %454 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_109 = stablehlo.constant dense<0> : tensor<i32>
    %456 = call @_where_4(%453, %455, %c_109) : (tensor<1xi1>, tensor<1xf32>, tensor<i32>) -> tensor<1xf32>
    %457 = stablehlo.slice %124 [0:1, 1:2] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %458 = stablehlo.reshape %457 : (tensor<1x1xf32>) -> tensor<1xf32>
    %459 = stablehlo.multiply %456, %458 : tensor<1xf32>
    %460 = stablehlo.optimization_barrier %459 : tensor<1xf32>
    %461 = stablehlo.add %444, %460 : tensor<1xf32>
    %462 = stablehlo.optimization_barrier %461 : tensor<1xf32>
    %463 = stablehlo.add %447, %456 : tensor<1xf32>
    %464 = stablehlo.optimization_barrier %463 : tensor<1xf32>
    %c_110 = stablehlo.constant dense<1> : tensor<i32>
    %465 = stablehlo.broadcast_in_dim %c_110, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %466 = stablehlo.subtract %426, %465 : tensor<1xi32>
    %c_111 = stablehlo.constant dense<0> : tensor<i32>
    %467 = stablehlo.broadcast_in_dim %c_111, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %468 = stablehlo.maximum %466, %467 : tensor<1xi32>
    %c_112 = stablehlo.constant dense<2> : tensor<i32>
    %469 = stablehlo.broadcast_in_dim %c_112, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %470 = stablehlo.compare LE, %468, %469, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %471 = stablehlo.slice %arg0 [0:1, 2:3, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %472 = stablehlo.reshape %471 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_113 = stablehlo.constant dense<0> : tensor<i32>
    %473 = call @_where_4(%470, %472, %c_113) : (tensor<1xi1>, tensor<1xf32>, tensor<i32>) -> tensor<1xf32>
    %474 = stablehlo.slice %124 [0:1, 2:3] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %475 = stablehlo.reshape %474 : (tensor<1x1xf32>) -> tensor<1xf32>
    %476 = stablehlo.multiply %473, %475 : tensor<1xf32>
    %477 = stablehlo.optimization_barrier %476 : tensor<1xf32>
    %478 = stablehlo.add %462, %477 : tensor<1xf32>
    %479 = stablehlo.optimization_barrier %478 : tensor<1xf32>
    %480 = stablehlo.add %464, %473 : tensor<1xf32>
    %481 = stablehlo.optimization_barrier %480 : tensor<1xf32>
    %c_114 = stablehlo.constant dense<1> : tensor<i32>
    %482 = stablehlo.broadcast_in_dim %c_114, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %483 = stablehlo.subtract %426, %482 : tensor<1xi32>
    %c_115 = stablehlo.constant dense<0> : tensor<i32>
    %484 = stablehlo.broadcast_in_dim %c_115, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %485 = stablehlo.maximum %483, %484 : tensor<1xi32>
    %c_116 = stablehlo.constant dense<3> : tensor<i32>
    %486 = stablehlo.broadcast_in_dim %c_116, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %487 = stablehlo.compare LE, %485, %486, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %488 = stablehlo.slice %arg0 [0:1, 3:4, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %489 = stablehlo.reshape %488 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_117 = stablehlo.constant dense<0> : tensor<i32>
    %490 = call @_where_4(%487, %489, %c_117) : (tensor<1xi1>, tensor<1xf32>, tensor<i32>) -> tensor<1xf32>
    %491 = stablehlo.slice %124 [0:1, 3:4] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %492 = stablehlo.reshape %491 : (tensor<1x1xf32>) -> tensor<1xf32>
    %493 = stablehlo.multiply %490, %492 : tensor<1xf32>
    %494 = stablehlo.optimization_barrier %493 : tensor<1xf32>
    %495 = stablehlo.add %479, %494 : tensor<1xf32>
    %496 = stablehlo.optimization_barrier %495 : tensor<1xf32>
    %497 = stablehlo.add %481, %490 : tensor<1xf32>
    %498 = stablehlo.optimization_barrier %497 : tensor<1xf32>
    %cst_118 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %499 = stablehlo.broadcast_in_dim %cst_118, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %500 = stablehlo.compare EQ, %498, %499, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %c_119 = stablehlo.constant dense<1> : tensor<i32>
    %501 = call @_where_1(%500, %c_119, %498) : (tensor<1xi1>, tensor<i32>, tensor<1xf32>) -> tensor<1xf32>
    %502 = stablehlo.divide %496, %501 : tensor<1xf32>
    %503 = stablehlo.optimization_barrier %502 : tensor<1xf32>
    %504 = stablehlo.slice %arg1 [0:1, 11:12] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %505 = stablehlo.reshape %504 : (tensor<1x1xf32>) -> tensor<1xf32>
    %506 = stablehlo.negate %505 : tensor<1xf32>
    %507 = stablehlo.multiply %506, %503 : tensor<1xf32>
    %508 = stablehlo.optimization_barrier %507 : tensor<1xf32>
    %cst_120 = stablehlo.constant dense<1.000000e+01> : tensor<f32>
    %509 = stablehlo.broadcast_in_dim %cst_120, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %510 = stablehlo.power %509, %508 : tensor<1xf32>
    %cst_121 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %511 = stablehlo.broadcast_in_dim %cst_121, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %512 = stablehlo.compare EQ, %428, %511, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %c_122 = stablehlo.constant dense<1> : tensor<i32>
    %513 = call @_where_1(%512, %c_122, %510) : (tensor<1xi1>, tensor<i32>, tensor<1xf32>) -> tensor<1xf32>
    %514 = stablehlo.slice %arg1 [0:1, 6:7] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %515 = stablehlo.reshape %514 : (tensor<1x1xf32>) -> tensor<1xf32>
    %516 = stablehlo.slice %arg1 [0:1, 5:6] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %517 = stablehlo.reshape %516 : (tensor<1x1xf32>) -> tensor<1xf32>
    %518 = stablehlo.tan %517 : tensor<1xf32>
    %519 = stablehlo.multiply %515, %518 : tensor<1xf32>
    %520 = stablehlo.optimization_barrier %519 : tensor<1xf32>
    %521 = stablehlo.multiply %520, %428 : tensor<1xf32>
    %522 = stablehlo.optimization_barrier %521 : tensor<1xf32>
    %523 = stablehlo.multiply %522, %513 : tensor<1xf32>
    %524 = stablehlo.optimization_barrier %523 : tensor<1xf32>
    %525 = stablehlo.broadcast_in_dim %437, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %526 = stablehlo.broadcast_in_dim %456, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %527 = stablehlo.broadcast_in_dim %473, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %528 = stablehlo.broadcast_in_dim %490, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %529 = stablehlo.concatenate %525, %526, %527, %528, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<4x1xf32>
    %530 = stablehlo.slice %62 [0:1, 1:5] : (tensor<1x5xf32>) -> tensor<1x4xf32>
    %531 = stablehlo.broadcast_in_dim %418, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %532 = stablehlo.broadcast_in_dim %531, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
    %533 = stablehlo.compare LT, %530, %532, FLOAT : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xi1>
    %534 = stablehlo.convert %533 : (tensor<1x4xi1>) -> tensor<1x4xi32>
    %c_123 = stablehlo.constant dense<0> : tensor<i32>
    %535 = stablehlo.reduce(%534 init: %c_123) applies stablehlo.add across dimensions = [1] : (tensor<1x4xi32>, tensor<i32>) -> tensor<1xi32>
    %536 = stablehlo.slice %62 [0:1, 1:5] : (tensor<1x5xf32>) -> tensor<1x4xf32>
    %537 = stablehlo.broadcast_in_dim %303, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %538 = stablehlo.broadcast_in_dim %537, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x4xf32>
    %539 = stablehlo.compare LT, %536, %538, FLOAT : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xi1>
    %540 = stablehlo.convert %539 : (tensor<1x4xi1>) -> tensor<1x4xi32>
    %c_124 = stablehlo.constant dense<0> : tensor<i32>
    %541 = stablehlo.reduce(%540 init: %c_124) applies stablehlo.add across dimensions = [1] : (tensor<1x4xi32>, tensor<i32>) -> tensor<1xi32>
    %542 = stablehlo.subtract %303, %418 : tensor<1xf32>
    %543 = stablehlo.optimization_barrier %542 : tensor<1xf32>
    %c_125 = stablehlo.constant dense<0> : tensor<i32>
    %544 = stablehlo.broadcast_in_dim %c_125, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %545 = stablehlo.compare LE, %535, %544, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_126 = stablehlo.constant dense<0> : tensor<i32>
    %546 = stablehlo.broadcast_in_dim %c_126, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %547 = stablehlo.compare GE, %541, %546, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %548 = stablehlo.and %545, %547 : tensor<1xi1>
    %549 = stablehlo.slice %arg0 [0:1, 0:1, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %550 = stablehlo.reshape %549 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_127 = stablehlo.constant dense<0> : tensor<i32>
    %551 = call @_where_4(%548, %550, %c_127) : (tensor<1xi1>, tensor<1xf32>, tensor<i32>) -> tensor<1xf32>
    %552 = stablehlo.slice %arg1 [0:1, 11:12] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %553 = stablehlo.reshape %552 : (tensor<1x1xf32>) -> tensor<1xf32>
    %554 = stablehlo.negate %553 : tensor<1xf32>
    %555 = stablehlo.slice %124 [0:1, 0:1] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %556 = stablehlo.reshape %555 : (tensor<1x1xf32>) -> tensor<1xf32>
    %557 = stablehlo.slice %124 [0:1, 1:2] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %558 = stablehlo.reshape %557 : (tensor<1x1xf32>) -> tensor<1xf32>
    %559 = stablehlo.add %556, %558 : tensor<1xf32>
    %560 = stablehlo.optimization_barrier %559 : tensor<1xf32>
    %cst_128 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %561 = stablehlo.broadcast_in_dim %cst_128, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %562 = stablehlo.multiply %561, %560 : tensor<1xf32>
    %563 = stablehlo.optimization_barrier %562 : tensor<1xf32>
    %564 = stablehlo.multiply %554, %563 : tensor<1xf32>
    %565 = stablehlo.optimization_barrier %564 : tensor<1xf32>
    %cst_129 = stablehlo.constant dense<1.000000e+01> : tensor<f32>
    %566 = stablehlo.broadcast_in_dim %cst_129, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %567 = stablehlo.power %566, %565 : tensor<1xf32>
    %568 = stablehlo.multiply %551, %567 : tensor<1xf32>
    %569 = stablehlo.optimization_barrier %568 : tensor<1xf32>
    %cst_130 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %570 = stablehlo.broadcast_in_dim %cst_130, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %571 = stablehlo.add %570, %569 : tensor<1xf32>
    %572 = stablehlo.optimization_barrier %571 : tensor<1xf32>
    %cst_131 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %573 = stablehlo.broadcast_in_dim %cst_131, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %574 = stablehlo.add %573, %551 : tensor<1xf32>
    %575 = stablehlo.optimization_barrier %574 : tensor<1xf32>
    %c_132 = stablehlo.constant dense<1> : tensor<i32>
    %576 = stablehlo.broadcast_in_dim %c_132, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %577 = stablehlo.compare LE, %535, %576, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_133 = stablehlo.constant dense<1> : tensor<i32>
    %578 = stablehlo.broadcast_in_dim %c_133, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %579 = stablehlo.compare GE, %541, %578, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %580 = stablehlo.and %577, %579 : tensor<1xi1>
    %581 = stablehlo.slice %arg0 [0:1, 1:2, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %582 = stablehlo.reshape %581 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_134 = stablehlo.constant dense<0> : tensor<i32>
    %583 = call @_where_4(%580, %582, %c_134) : (tensor<1xi1>, tensor<1xf32>, tensor<i32>) -> tensor<1xf32>
    %584 = stablehlo.slice %arg1 [0:1, 11:12] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %585 = stablehlo.reshape %584 : (tensor<1x1xf32>) -> tensor<1xf32>
    %586 = stablehlo.negate %585 : tensor<1xf32>
    %587 = stablehlo.slice %124 [0:1, 1:2] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %588 = stablehlo.reshape %587 : (tensor<1x1xf32>) -> tensor<1xf32>
    %589 = stablehlo.slice %124 [0:1, 2:3] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %590 = stablehlo.reshape %589 : (tensor<1x1xf32>) -> tensor<1xf32>
    %591 = stablehlo.add %588, %590 : tensor<1xf32>
    %592 = stablehlo.optimization_barrier %591 : tensor<1xf32>
    %cst_135 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %593 = stablehlo.broadcast_in_dim %cst_135, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %594 = stablehlo.multiply %593, %592 : tensor<1xf32>
    %595 = stablehlo.optimization_barrier %594 : tensor<1xf32>
    %596 = stablehlo.multiply %586, %595 : tensor<1xf32>
    %597 = stablehlo.optimization_barrier %596 : tensor<1xf32>
    %cst_136 = stablehlo.constant dense<1.000000e+01> : tensor<f32>
    %598 = stablehlo.broadcast_in_dim %cst_136, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %599 = stablehlo.power %598, %597 : tensor<1xf32>
    %600 = stablehlo.multiply %583, %599 : tensor<1xf32>
    %601 = stablehlo.optimization_barrier %600 : tensor<1xf32>
    %602 = stablehlo.add %572, %601 : tensor<1xf32>
    %603 = stablehlo.optimization_barrier %602 : tensor<1xf32>
    %604 = stablehlo.add %575, %583 : tensor<1xf32>
    %605 = stablehlo.optimization_barrier %604 : tensor<1xf32>
    %c_137 = stablehlo.constant dense<2> : tensor<i32>
    %606 = stablehlo.broadcast_in_dim %c_137, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %607 = stablehlo.compare LE, %535, %606, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_138 = stablehlo.constant dense<2> : tensor<i32>
    %608 = stablehlo.broadcast_in_dim %c_138, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %609 = stablehlo.compare GE, %541, %608, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %610 = stablehlo.and %607, %609 : tensor<1xi1>
    %611 = stablehlo.slice %arg0 [0:1, 2:3, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %612 = stablehlo.reshape %611 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_139 = stablehlo.constant dense<0> : tensor<i32>
    %613 = call @_where_4(%610, %612, %c_139) : (tensor<1xi1>, tensor<1xf32>, tensor<i32>) -> tensor<1xf32>
    %614 = stablehlo.slice %arg1 [0:1, 11:12] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %615 = stablehlo.reshape %614 : (tensor<1x1xf32>) -> tensor<1xf32>
    %616 = stablehlo.negate %615 : tensor<1xf32>
    %617 = stablehlo.slice %124 [0:1, 2:3] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %618 = stablehlo.reshape %617 : (tensor<1x1xf32>) -> tensor<1xf32>
    %619 = stablehlo.slice %124 [0:1, 3:4] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %620 = stablehlo.reshape %619 : (tensor<1x1xf32>) -> tensor<1xf32>
    %621 = stablehlo.add %618, %620 : tensor<1xf32>
    %622 = stablehlo.optimization_barrier %621 : tensor<1xf32>
    %cst_140 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %623 = stablehlo.broadcast_in_dim %cst_140, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %624 = stablehlo.multiply %623, %622 : tensor<1xf32>
    %625 = stablehlo.optimization_barrier %624 : tensor<1xf32>
    %626 = stablehlo.multiply %616, %625 : tensor<1xf32>
    %627 = stablehlo.optimization_barrier %626 : tensor<1xf32>
    %cst_141 = stablehlo.constant dense<1.000000e+01> : tensor<f32>
    %628 = stablehlo.broadcast_in_dim %cst_141, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %629 = stablehlo.power %628, %627 : tensor<1xf32>
    %630 = stablehlo.multiply %613, %629 : tensor<1xf32>
    %631 = stablehlo.optimization_barrier %630 : tensor<1xf32>
    %632 = stablehlo.add %603, %631 : tensor<1xf32>
    %633 = stablehlo.optimization_barrier %632 : tensor<1xf32>
    %634 = stablehlo.add %605, %613 : tensor<1xf32>
    %635 = stablehlo.optimization_barrier %634 : tensor<1xf32>
    %c_142 = stablehlo.constant dense<3> : tensor<i32>
    %636 = stablehlo.broadcast_in_dim %c_142, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %637 = stablehlo.compare LE, %535, %636, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_143 = stablehlo.constant dense<3> : tensor<i32>
    %638 = stablehlo.broadcast_in_dim %c_143, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %639 = stablehlo.compare GE, %541, %638, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %640 = stablehlo.and %637, %639 : tensor<1xi1>
    %641 = stablehlo.slice %arg0 [0:1, 3:4, 2:3] : (tensor<1x4x6xf32>) -> tensor<1x1x1xf32>
    %642 = stablehlo.reshape %641 : (tensor<1x1x1xf32>) -> tensor<1xf32>
    %c_144 = stablehlo.constant dense<0> : tensor<i32>
    %643 = call @_where_4(%640, %642, %c_144) : (tensor<1xi1>, tensor<1xf32>, tensor<i32>) -> tensor<1xf32>
    %644 = stablehlo.slice %arg1 [0:1, 11:12] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %645 = stablehlo.reshape %644 : (tensor<1x1xf32>) -> tensor<1xf32>
    %646 = stablehlo.negate %645 : tensor<1xf32>
    %647 = stablehlo.slice %124 [0:1, 3:4] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %648 = stablehlo.reshape %647 : (tensor<1x1xf32>) -> tensor<1xf32>
    %649 = stablehlo.slice %124 [0:1, 3:4] : (tensor<1x4xf32>) -> tensor<1x1xf32>
    %650 = stablehlo.reshape %649 : (tensor<1x1xf32>) -> tensor<1xf32>
    %651 = stablehlo.add %648, %650 : tensor<1xf32>
    %652 = stablehlo.optimization_barrier %651 : tensor<1xf32>
    %cst_145 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %653 = stablehlo.broadcast_in_dim %cst_145, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %654 = stablehlo.multiply %653, %652 : tensor<1xf32>
    %655 = stablehlo.optimization_barrier %654 : tensor<1xf32>
    %656 = stablehlo.multiply %646, %655 : tensor<1xf32>
    %657 = stablehlo.optimization_barrier %656 : tensor<1xf32>
    %cst_146 = stablehlo.constant dense<1.000000e+01> : tensor<f32>
    %658 = stablehlo.broadcast_in_dim %cst_146, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %659 = stablehlo.power %658, %657 : tensor<1xf32>
    %660 = stablehlo.multiply %643, %659 : tensor<1xf32>
    %661 = stablehlo.optimization_barrier %660 : tensor<1xf32>
    %662 = stablehlo.add %633, %661 : tensor<1xf32>
    %663 = stablehlo.optimization_barrier %662 : tensor<1xf32>
    %664 = stablehlo.add %635, %643 : tensor<1xf32>
    %665 = stablehlo.optimization_barrier %664 : tensor<1xf32>
    %cst_147 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %666 = stablehlo.broadcast_in_dim %cst_147, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %667 = stablehlo.compare EQ, %665, %666, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %c_148 = stablehlo.constant dense<1> : tensor<i32>
    %668 = call @_where_1(%667, %c_148, %665) : (tensor<1xi1>, tensor<i32>, tensor<1xf32>) -> tensor<1xf32>
    %669 = stablehlo.divide %663, %668 : tensor<1xf32>
    %670 = stablehlo.optimization_barrier %669 : tensor<1xf32>
    %cst_149 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %671 = stablehlo.broadcast_in_dim %cst_149, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %672 = stablehlo.compare EQ, %543, %671, FLOAT : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %c_150 = stablehlo.constant dense<1> : tensor<i32>
    %673 = call @_where_1(%672, %c_150, %670) : (tensor<1xi1>, tensor<i32>, tensor<1xf32>) -> tensor<1xf32>
    %674 = stablehlo.slice %arg1 [0:1, 7:8] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %675 = stablehlo.reshape %674 : (tensor<1x1xf32>) -> tensor<1xf32>
    %cst_151 = stablehlo.constant dense<1.000000e+03> : tensor<f32>
    %676 = stablehlo.broadcast_in_dim %cst_151, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %677 = stablehlo.multiply %676, %675 : tensor<1xf32>
    %678 = stablehlo.optimization_barrier %677 : tensor<1xf32>
    %679 = stablehlo.slice %arg1 [0:1, 5:6] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %680 = stablehlo.reshape %679 : (tensor<1x1xf32>) -> tensor<1xf32>
    %681 = stablehlo.sine %680 : tensor<1xf32>
    %682 = stablehlo.multiply %678, %681 : tensor<1xf32>
    %683 = stablehlo.optimization_barrier %682 : tensor<1xf32>
    %684 = stablehlo.slice %arg1 [0:1, 3:4] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %685 = stablehlo.reshape %684 : (tensor<1x1xf32>) -> tensor<1xf32>
    %686 = stablehlo.multiply %685, %673 : tensor<1xf32>
    %687 = stablehlo.optimization_barrier %686 : tensor<1xf32>
    %688 = stablehlo.multiply %683, %687 : tensor<1xf32>
    %689 = stablehlo.optimization_barrier %688 : tensor<1xf32>
    %690 = stablehlo.multiply %689, %543 : tensor<1xf32>
    %691 = stablehlo.optimization_barrier %690 : tensor<1xf32>
    %692 = stablehlo.compare GE, %535, %541, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %c_152 = stablehlo.constant dense<0> : tensor<i32>
    %693 = call @_where_1(%692, %c_152, %691) : (tensor<1xi1>, tensor<i32>, tensor<1xf32>) -> tensor<1xf32>
    %694 = stablehlo.broadcast_in_dim %551, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %695 = stablehlo.broadcast_in_dim %583, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %696 = stablehlo.broadcast_in_dim %613, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %697 = stablehlo.broadcast_in_dim %643, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %698 = stablehlo.concatenate %694, %695, %696, %697, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<4x1xf32>
    %c_153 = stablehlo.constant dense<0> : tensor<i32>
    %699 = call @_where_4(%417, %693, %c_153) : (tensor<1xi1>, tensor<1xf32>, tensor<i32>) -> tensor<1xf32>
    %c_154 = stablehlo.constant dense<1> : tensor<i32>
    %700 = call @_where_4(%417, %673, %c_154) : (tensor<1xi1>, tensor<1xf32>, tensor<i32>) -> tensor<1xf32>
    %c_155 = stablehlo.constant dense<0> : tensor<i32>
    %701 = call @_where_5(%417, %698, %c_155) : (tensor<1xi1>, tensor<4x1xf32>, tensor<i32>) -> tensor<1x4xf32>
    %c_156 = stablehlo.constant dense<0> : tensor<i32>
    %702 = stablehlo.broadcast_in_dim %c_156, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %703 = stablehlo.compare GE, %288, %702, SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %704 = stablehlo.convert %703 : (tensor<1xi1>) -> tensor<1xf32>
    %705 = stablehlo.convert %417 : (tensor<1xi1>) -> tensor<1xf32>
    %706 = stablehlo.convert %290 : (tensor<1xi32>) -> tensor<1xf32>
    %707 = stablehlo.convert %85 : (tensor<1xi32>) -> tensor<1xf32>
    %708 = stablehlo.slice %arg1 [0:1, 9:10] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %709 = stablehlo.reshape %708 : (tensor<1x1xf32>) -> tensor<1xf32>
    %710 = stablehlo.slice %arg1 [0:1, 0:1] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %711 = stablehlo.reshape %710 : (tensor<1x1xf32>) -> tensor<1xf32>
    %712 = stablehlo.negate %711 : tensor<1xf32>
    %713 = stablehlo.slice %arg1 [0:1, 1:2] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %714 = stablehlo.reshape %713 : (tensor<1x1xf32>) -> tensor<1xf32>
    %715 = stablehlo.divide %235, %714 : tensor<1xf32>
    %716 = stablehlo.optimization_barrier %715 : tensor<1xf32>
    %717 = stablehlo.log_plus_one %716 : tensor<1xf32>
    %718 = stablehlo.negate %717 : tensor<1xf32>
    %719 = stablehlo.slice %arg1 [0:1, 2:3] : (tensor<1x12xf32>) -> tensor<1x1xf32>
    %720 = stablehlo.reshape %719 : (tensor<1x1xf32>) -> tensor<1xf32>
    %721 = stablehlo.divide %718, %720 : tensor<1xf32>
    %722 = stablehlo.optimization_barrier %721 : tensor<1xf32>
    %723 = stablehlo.exponential_minus_one %722 : tensor<1xf32>
    %724 = stablehlo.multiply %712, %723 : tensor<1xf32>
    %725 = stablehlo.optimization_barrier %724 : tensor<1xf32>
    %726 = stablehlo.maximum %709, %725 : tensor<1xf32>
    %c_157 = stablehlo.constant dense<1> : tensor<i32>
    %727 = stablehlo.broadcast_in_dim %c_157, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %728 = stablehlo.add %535, %727 : tensor<1xi32>
    %c_158 = stablehlo.constant dense<-1> : tensor<i32>
    %729 = call @_where_6(%417, %728, %c_158) : (tensor<1xi1>, tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
    %730 = stablehlo.convert %729 : (tensor<1xi32>) -> tensor<1xf32>
    %731 = stablehlo.broadcast_in_dim %235, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %732 = stablehlo.broadcast_in_dim %303, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %733 = stablehlo.broadcast_in_dim %418, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %734 = stablehlo.broadcast_in_dim %704, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %735 = stablehlo.broadcast_in_dim %705, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %736 = stablehlo.broadcast_in_dim %706, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %737 = stablehlo.broadcast_in_dim %707, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %738 = stablehlo.broadcast_in_dim %524, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %739 = stablehlo.broadcast_in_dim %699, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %740 = stablehlo.broadcast_in_dim %513, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %741 = stablehlo.broadcast_in_dim %700, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %742 = stablehlo.broadcast_in_dim %726, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %743 = stablehlo.broadcast_in_dim %730, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %744 = stablehlo.concatenate %731, %732, %733, %734, %735, %736, %737, %738, %739, %740, %741, %742, %743, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<13x1xf32>
    %745 = stablehlo.transpose %744, dims = [1, 0] : (tensor<13x1xf32>) -> tensor<1x13xf32>
    %746 = stablehlo.transpose %529, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %747 = stablehlo.concatenate %745, %746, %701, %124, %107, dim = 1 : (tensor<1x13xf32>, tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x29xf32>
    %748 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %749 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %750 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %751 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %752 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %753 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %754 = stablehlo.broadcast_in_dim %32, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %755 = stablehlo.broadcast_in_dim %36, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %756 = stablehlo.broadcast_in_dim %38, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<1x4x1xf32>
    %757 = stablehlo.concatenate %748, %749, %750, %751, %752, %753, %754, %755, %756, dim = 2 : (tensor<1x4x1xf32>, tensor<1x4x1xf32>, tensor<1x4x1xf32>, tensor<1x4x1xf32>, tensor<1x4x1xf32>, tensor<1x4x1xf32>, tensor<1x4x1xf32>, tensor<1x4x1xf32>, tensor<1x4x1xf32>) -> tensor<1x4x9xf32>
    return %747, %757 : tensor<1x29xf32>, tensor<1x4x9xf32>
  }
  func.func private @_where(%arg0: tensor<1xi1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<1xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.select %arg0, %2, %1 : tensor<1xi1>, tensor<1xi32>
    return %3 : tensor<1xi32>
  }
  func.func private @_where_0(%arg0: tensor<1xi1>, %arg1: tensor<i32>, %arg2: tensor<1xi32>) -> tensor<1xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<1xi1>, tensor<1xi32>
    return %2 : tensor<1xi32>
  }
  func.func private @clip(%arg0: tensor<1xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<1xi32> {
    %0 = stablehlo.convert %arg1 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1xi32>
    %3 = stablehlo.convert %arg2 : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %5 = stablehlo.minimum %4, %2 : tensor<1xi32>
    return %5 : tensor<1xi32>
  }
  func.func private @_where_1(%arg0: tensor<1xi1>, %arg1: tensor<i32>, %arg2: tensor<1xf32>) -> tensor<1xf32> {
    %0 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %2 = stablehlo.select %arg0, %1, %arg2 : tensor<1xi1>, tensor<1xf32>
    return %2 : tensor<1xf32>
  }
  func.func private @clip_2(%arg0: tensor<1xf32>, %arg1: tensor<i32>, %arg2: tensor<1xf32>) -> tensor<1xf32> {
    %0 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %2 = stablehlo.maximum %1, %arg0 : tensor<1xf32>
    %3 = stablehlo.minimum %arg2, %2 : tensor<1xf32>
    return %3 : tensor<1xf32>
  }
  func.func private @_where_3(%arg0: tensor<1xi1>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1xf32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1xi1>, tensor<1xf32>
    return %0 : tensor<1xf32>
  }
  func.func private @_where_4(%arg0: tensor<1xi1>, %arg1: tensor<1xf32>, %arg2: tensor<i32>) -> tensor<1xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<1xi1>, tensor<1xf32>
    return %2 : tensor<1xf32>
  }
  func.func private @_where_5(%arg0: tensor<1xi1>, %arg1: tensor<4x1xf32>, %arg2: tensor<i32>) -> tensor<1x4xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
    %3 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<4x1xf32>) -> tensor<1x4xf32>
    %4 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<1xi1>) -> tensor<1x4xi1>
    %5 = stablehlo.select %4, %3, %2 : tensor<1x4xi1>, tensor<1x4xf32>
    return %5 : tensor<1x4xf32>
  }
  func.func private @_where_6(%arg0: tensor<1xi1>, %arg1: tensor<1xi32>, %arg2: tensor<i32>) -> tensor<1xi32> {
    %0 = stablehlo.convert %arg2 : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<1xi1>, tensor<1xi32>
    return %2 : tensor<1xi32>
  }
}
