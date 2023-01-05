; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i64 @__float_add(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8)

declare i64 @__float_mul(i64, i64, i8, i8, i32, i1, i1, i1, i1, i8)

declare i64 @__float_cast(i64, i8, i8, i32, i1, i1, i1, i1, i8, i8, i8, i32, i1, i1, i1, i1, i8)

define i64 @cast_float(double %0) {
  %2 = bitcast double %0 to i64
  %3 = call i64 @__float_cast(i64 %2, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1)
  ret i64 %3
}

define i64 @_mlir_ciface_cast_float(double %0) {
  %2 = call i64 @cast_float(double %0)
  ret i64 %2
}

define void @kernel_sf(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, ptr %25, ptr %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, ptr %34, ptr %35, i64 %36, i64 %37, i64 %38, i64 %39, i64 %40, i64 %41, i64 %42, ptr %43, ptr %44, i64 %45, i64 %46, i64 %47, i64 %48, i64 %49, i64 %50, i64 %51, ptr %52, ptr %53, i64 %54, i64 %55, i64 %56, i64 %57, i64 %58, i64 %59, i64 %60, ptr %61, ptr %62, i64 %63, i64 %64, i64 %65, i64 %66, i64 %67, i64 %68, i64 %69, ptr %70, ptr %71, i64 %72, i64 %73, i64 %74, i64 %75, i64 %76, i64 %77, i64 %78, ptr %79, ptr %80, i64 %81, i64 %82, i64 %83, i64 %84, i64 %85, i64 %86, i64 %87) {
  %89 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %0, 0
  %90 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %89, ptr %1, 1
  %91 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %90, i64 %2, 2
  %92 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %91, i64 %3, 3, 0
  %93 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %92, i64 %5, 4, 0
  %94 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %93, i64 %4, 3, 1
  %95 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %94, i64 %6, 4, 1
  %96 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %7, 0
  %97 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %96, ptr %8, 1
  %98 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %97, i64 %9, 2
  %99 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %98, i64 %10, 3, 0
  %100 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %99, i64 %13, 4, 0
  %101 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %100, i64 %11, 3, 1
  %102 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %101, i64 %14, 4, 1
  %103 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %102, i64 %12, 3, 2
  %104 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %103, i64 %15, 4, 2
  %105 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %16, 0
  %106 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %105, ptr %17, 1
  %107 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %106, i64 %18, 2
  %108 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %107, i64 %19, 3, 0
  %109 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %108, i64 %22, 4, 0
  %110 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %109, i64 %20, 3, 1
  %111 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %110, i64 %23, 4, 1
  %112 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %111, i64 %21, 3, 2
  %113 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %112, i64 %24, 4, 2
  %114 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %25, 0
  %115 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %114, ptr %26, 1
  %116 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %115, i64 %27, 2
  %117 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %116, i64 %28, 3, 0
  %118 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %117, i64 %31, 4, 0
  %119 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %118, i64 %29, 3, 1
  %120 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %119, i64 %32, 4, 1
  %121 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %120, i64 %30, 3, 2
  %122 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %121, i64 %33, 4, 2
  %123 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %34, 0
  %124 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %123, ptr %35, 1
  %125 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %124, i64 %36, 2
  %126 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %125, i64 %37, 3, 0
  %127 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %126, i64 %40, 4, 0
  %128 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %127, i64 %38, 3, 1
  %129 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %128, i64 %41, 4, 1
  %130 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %129, i64 %39, 3, 2
  %131 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %130, i64 %42, 4, 2
  %132 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %43, 0
  %133 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %132, ptr %44, 1
  %134 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %133, i64 %45, 2
  %135 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %134, i64 %46, 3, 0
  %136 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %135, i64 %49, 4, 0
  %137 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %136, i64 %47, 3, 1
  %138 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %137, i64 %50, 4, 1
  %139 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %138, i64 %48, 3, 2
  %140 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %139, i64 %51, 4, 2
  %141 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %52, 0
  %142 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %141, ptr %53, 1
  %143 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %142, i64 %54, 2
  %144 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %143, i64 %55, 3, 0
  %145 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %144, i64 %58, 4, 0
  %146 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %145, i64 %56, 3, 1
  %147 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %146, i64 %59, 4, 1
  %148 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %147, i64 %57, 3, 2
  %149 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %148, i64 %60, 4, 2
  %150 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %61, 0
  %151 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %150, ptr %62, 1
  %152 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %151, i64 %63, 2
  %153 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %152, i64 %64, 3, 0
  %154 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %153, i64 %67, 4, 0
  %155 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %154, i64 %65, 3, 1
  %156 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %155, i64 %68, 4, 1
  %157 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %156, i64 %66, 3, 2
  %158 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %157, i64 %69, 4, 2
  %159 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %70, 0
  %160 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %159, ptr %71, 1
  %161 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %160, i64 %72, 2
  %162 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %161, i64 %73, 3, 0
  %163 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %162, i64 %76, 4, 0
  %164 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %163, i64 %74, 3, 1
  %165 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %164, i64 %77, 4, 1
  %166 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %165, i64 %75, 3, 2
  %167 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %166, i64 %78, 4, 2
  %168 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %79, 0
  %169 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %168, ptr %80, 1
  %170 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %169, i64 %81, 2
  %171 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %170, i64 %82, 3, 0
  %172 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %171, i64 %85, 4, 0
  %173 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %172, i64 %83, 3, 1
  %174 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %173, i64 %86, 4, 1
  %175 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %174, i64 %84, 3, 2
  %176 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %175, i64 %87, 4, 2
  %177 = call i64 @cast_float(double 0.000000e+00)
  br label %178

178:                                              ; preds = %274, %88
  %179 = phi i64 [ %275, %274 ], [ 0, %88 ]
  %180 = icmp slt i64 %179, 11
  br i1 %180, label %181, label %276

181:                                              ; preds = %178
  br label %182

182:                                              ; preds = %272, %181
  %183 = phi i64 [ %273, %272 ], [ 0, %181 ]
  %184 = icmp slt i64 %183, 11
  br i1 %184, label %185, label %274

185:                                              ; preds = %182
  br label %186

186:                                              ; preds = %270, %185
  %187 = phi i64 [ %271, %270 ], [ 0, %185 ]
  %188 = icmp slt i64 %187, 11
  br i1 %188, label %189, label %272

189:                                              ; preds = %186
  %190 = mul i64 %179, 121
  %191 = mul i64 %183, 11
  %192 = add i64 %190, %191
  %193 = add i64 %192, %187
  %194 = getelementptr i64, ptr %62, i64 %193
  store i64 %177, ptr %194, align 4
  br label %195

195:                                              ; preds = %268, %189
  %196 = phi i64 [ %269, %268 ], [ 0, %189 ]
  %197 = icmp slt i64 %196, 21
  br i1 %197, label %198, label %270

198:                                              ; preds = %195
  %199 = mul i64 %196, -1
  %200 = add i64 %199, 10
  %201 = icmp sge i64 %200, 0
  br i1 %201, label %202, label %226

202:                                              ; preds = %198
  %203 = mul i64 %179, 11
  %204 = add i64 %203, %196
  %205 = getelementptr i64, ptr %1, i64 %204
  %206 = load i64, ptr %205, align 4
  %207 = mul i64 %183, 121
  %208 = mul i64 %187, 11
  %209 = add i64 %207, %208
  %210 = add i64 %209, %196
  %211 = getelementptr i64, ptr %17, i64 %210
  %212 = load i64, ptr %211, align 4
  %213 = call i64 @__float_mul(i64 %206, i64 %212, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1)
  %214 = mul i64 %179, 121
  %215 = mul i64 %183, 11
  %216 = add i64 %214, %215
  %217 = add i64 %216, %187
  %218 = getelementptr i64, ptr %62, i64 %217
  %219 = load i64, ptr %218, align 4
  %220 = call i64 @__float_add(i64 %213, i64 %219, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1)
  %221 = mul i64 %179, 121
  %222 = mul i64 %183, 11
  %223 = add i64 %221, %222
  %224 = add i64 %223, %187
  %225 = getelementptr i64, ptr %62, i64 %224
  store i64 %220, ptr %225, align 4
  br label %226

226:                                              ; preds = %202, %198
  %227 = add i64 %196, -10
  %228 = icmp sge i64 %227, 0
  br i1 %228, label %229, label %256

229:                                              ; preds = %226
  %230 = add i64 %196, -10
  %231 = mul i64 %187, 11
  %232 = add i64 %231, %230
  %233 = getelementptr i64, ptr %1, i64 %232
  %234 = load i64, ptr %233, align 4
  %235 = mul i64 %179, 121
  %236 = mul i64 %183, 11
  %237 = add i64 %235, %236
  %238 = add i64 %237, %187
  %239 = getelementptr i64, ptr %62, i64 %238
  %240 = load i64, ptr %239, align 4
  %241 = call i64 @__float_mul(i64 %234, i64 %240, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1)
  %242 = add i64 %196, -10
  %243 = mul i64 %242, 121
  %244 = mul i64 %179, 11
  %245 = add i64 %243, %244
  %246 = add i64 %245, %183
  %247 = getelementptr i64, ptr %53, i64 %246
  %248 = load i64, ptr %247, align 4
  %249 = call i64 @__float_add(i64 %241, i64 %248, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1)
  %250 = add i64 %196, -10
  %251 = mul i64 %250, 121
  %252 = mul i64 %179, 11
  %253 = add i64 %251, %252
  %254 = add i64 %253, %183
  %255 = getelementptr i64, ptr %53, i64 %254
  store i64 %249, ptr %255, align 4
  br label %256

256:                                              ; preds = %229, %226
  %257 = icmp eq i64 %187, 0
  %258 = mul i64 %196, -1
  %259 = add i64 %258, 10
  %260 = icmp sge i64 %259, 0
  %261 = and i1 %257, %260
  br i1 %261, label %262, label %268

262:                                              ; preds = %256
  %263 = mul i64 %196, 121
  %264 = mul i64 %179, 11
  %265 = add i64 %263, %264
  %266 = add i64 %265, %183
  %267 = getelementptr i64, ptr %53, i64 %266
  store i64 %177, ptr %267, align 4
  br label %268

268:                                              ; preds = %262, %256
  %269 = add i64 %196, 1
  br label %195

270:                                              ; preds = %195
  %271 = add i64 %187, 1
  br label %186

272:                                              ; preds = %186
  %273 = add i64 %183, 1
  br label %182

274:                                              ; preds = %182
  %275 = add i64 %179, 1
  br label %178

276:                                              ; preds = %178
  br label %277

277:                                              ; preds = %395, %276
  %278 = phi i64 [ %396, %395 ], [ 0, %276 ]
  %279 = icmp slt i64 %278, 11
  br i1 %279, label %280, label %397

280:                                              ; preds = %277
  br label %281

281:                                              ; preds = %393, %280
  %282 = phi i64 [ %394, %393 ], [ 0, %280 ]
  %283 = icmp slt i64 %282, 11
  br i1 %283, label %284, label %395

284:                                              ; preds = %281
  br label %285

285:                                              ; preds = %391, %284
  %286 = phi i64 [ %392, %391 ], [ 0, %284 ]
  %287 = icmp slt i64 %286, 11
  br i1 %287, label %288, label %393

288:                                              ; preds = %285
  %289 = mul i64 %278, 121
  %290 = mul i64 %282, 11
  %291 = add i64 %289, %290
  %292 = add i64 %291, %286
  %293 = getelementptr i64, ptr %35, i64 %292
  store i64 %177, ptr %293, align 4
  br label %294

294:                                              ; preds = %389, %288
  %295 = phi i64 [ %390, %389 ], [ 0, %288 ]
  %296 = icmp slt i64 %295, 21
  br i1 %296, label %297, label %391

297:                                              ; preds = %294
  %298 = mul i64 %295, -1
  %299 = add i64 %298, 10
  %300 = icmp sge i64 %299, 0
  br i1 %300, label %301, label %347

301:                                              ; preds = %297
  %302 = mul i64 %278, 11
  %303 = add i64 %302, %295
  %304 = getelementptr i64, ptr %1, i64 %303
  %305 = load i64, ptr %304, align 4
  %306 = mul i64 %282, 121
  %307 = mul i64 %286, 11
  %308 = add i64 %306, %307
  %309 = add i64 %308, %295
  %310 = getelementptr i64, ptr %53, i64 %309
  %311 = load i64, ptr %310, align 4
  %312 = call i64 @__float_mul(i64 %305, i64 %311, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1)
  %313 = mul i64 %278, 121
  %314 = mul i64 %282, 11
  %315 = add i64 %313, %314
  %316 = add i64 %315, %286
  %317 = getelementptr i64, ptr %35, i64 %316
  %318 = load i64, ptr %317, align 4
  %319 = call i64 @__float_add(i64 %312, i64 %318, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1)
  %320 = mul i64 %278, 121
  %321 = mul i64 %282, 11
  %322 = add i64 %320, %321
  %323 = add i64 %322, %286
  %324 = getelementptr i64, ptr %35, i64 %323
  store i64 %319, ptr %324, align 4
  %325 = add i64 %295, -10
  %326 = icmp eq i64 %325, 0
  br i1 %326, label %327, label %346

327:                                              ; preds = %301
  %328 = mul i64 %278, 121
  %329 = mul i64 %282, 11
  %330 = add i64 %328, %329
  %331 = add i64 %330, %286
  %332 = getelementptr i64, ptr %8, i64 %331
  %333 = load i64, ptr %332, align 4
  %334 = mul i64 %278, 121
  %335 = mul i64 %282, 11
  %336 = add i64 %334, %335
  %337 = add i64 %336, %286
  %338 = getelementptr i64, ptr %35, i64 %337
  %339 = load i64, ptr %338, align 4
  %340 = call i64 @__float_mul(i64 %333, i64 %339, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1)
  %341 = mul i64 %278, 121
  %342 = mul i64 %282, 11
  %343 = add i64 %341, %342
  %344 = add i64 %343, %286
  %345 = getelementptr i64, ptr %44, i64 %344
  store i64 %340, ptr %345, align 4
  br label %346

346:                                              ; preds = %327, %301
  br label %347

347:                                              ; preds = %346, %297
  %348 = add i64 %295, -10
  %349 = icmp sge i64 %348, 0
  br i1 %349, label %350, label %377

350:                                              ; preds = %347
  %351 = add i64 %295, -10
  %352 = mul i64 %286, 11
  %353 = add i64 %352, %351
  %354 = getelementptr i64, ptr %1, i64 %353
  %355 = load i64, ptr %354, align 4
  %356 = mul i64 %278, 121
  %357 = mul i64 %282, 11
  %358 = add i64 %356, %357
  %359 = add i64 %358, %286
  %360 = getelementptr i64, ptr %35, i64 %359
  %361 = load i64, ptr %360, align 4
  %362 = call i64 @__float_mul(i64 %355, i64 %361, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1)
  %363 = add i64 %295, -10
  %364 = mul i64 %363, 121
  %365 = mul i64 %278, 11
  %366 = add i64 %364, %365
  %367 = add i64 %366, %282
  %368 = getelementptr i64, ptr %80, i64 %367
  %369 = load i64, ptr %368, align 4
  %370 = call i64 @__float_add(i64 %362, i64 %369, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1)
  %371 = add i64 %295, -10
  %372 = mul i64 %371, 121
  %373 = mul i64 %278, 11
  %374 = add i64 %372, %373
  %375 = add i64 %374, %282
  %376 = getelementptr i64, ptr %80, i64 %375
  store i64 %370, ptr %376, align 4
  br label %377

377:                                              ; preds = %350, %347
  %378 = icmp eq i64 %286, 0
  %379 = mul i64 %295, -1
  %380 = add i64 %379, 10
  %381 = icmp sge i64 %380, 0
  %382 = and i1 %378, %381
  br i1 %382, label %383, label %389

383:                                              ; preds = %377
  %384 = mul i64 %295, 121
  %385 = mul i64 %278, 11
  %386 = add i64 %384, %385
  %387 = add i64 %386, %282
  %388 = getelementptr i64, ptr %80, i64 %387
  store i64 %177, ptr %388, align 4
  br label %389

389:                                              ; preds = %383, %377
  %390 = add i64 %295, 1
  br label %294

391:                                              ; preds = %294
  %392 = add i64 %286, 1
  br label %285

393:                                              ; preds = %285
  %394 = add i64 %282, 1
  br label %281

395:                                              ; preds = %281
  %396 = add i64 %278, 1
  br label %277

397:                                              ; preds = %277
  br label %398

398:                                              ; preds = %492, %397
  %399 = phi i64 [ %493, %492 ], [ 0, %397 ]
  %400 = icmp slt i64 %399, 11
  br i1 %400, label %401, label %494

401:                                              ; preds = %398
  br label %402

402:                                              ; preds = %490, %401
  %403 = phi i64 [ %491, %490 ], [ 0, %401 ]
  %404 = icmp slt i64 %403, 11
  br i1 %404, label %405, label %492

405:                                              ; preds = %402
  br label %406

406:                                              ; preds = %488, %405
  %407 = phi i64 [ %489, %488 ], [ 0, %405 ]
  %408 = icmp slt i64 %407, 11
  br i1 %408, label %409, label %490

409:                                              ; preds = %406
  br label %410

410:                                              ; preds = %431, %409
  %411 = phi i64 [ %455, %431 ], [ 0, %409 ]
  %412 = icmp slt i64 %411, 11
  br i1 %412, label %413, label %456

413:                                              ; preds = %410
  %414 = add i64 %407, -10
  %415 = icmp eq i64 %414, 0
  br i1 %415, label %416, label %422

416:                                              ; preds = %413
  %417 = mul i64 %411, 121
  %418 = mul i64 %399, 11
  %419 = add i64 %417, %418
  %420 = add i64 %419, %403
  %421 = getelementptr i64, ptr %26, i64 %420
  store i64 %177, ptr %421, align 4
  br label %422

422:                                              ; preds = %416, %413
  %423 = add i64 %411, -10
  %424 = icmp eq i64 %423, 0
  br i1 %424, label %425, label %431

425:                                              ; preds = %422
  %426 = mul i64 %399, 121
  %427 = mul i64 %403, 11
  %428 = add i64 %426, %427
  %429 = add i64 %428, %407
  %430 = getelementptr i64, ptr %71, i64 %429
  store i64 %177, ptr %430, align 4
  br label %431

431:                                              ; preds = %425, %422
  %432 = mul i64 %411, 11
  %433 = add i64 %432, %399
  %434 = getelementptr i64, ptr %1, i64 %433
  %435 = load i64, ptr %434, align 4
  %436 = mul i64 %403, 121
  %437 = mul i64 %407, 11
  %438 = add i64 %436, %437
  %439 = add i64 %438, %411
  %440 = getelementptr i64, ptr %80, i64 %439
  %441 = load i64, ptr %440, align 4
  %442 = call i64 @__float_mul(i64 %435, i64 %441, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1)
  %443 = mul i64 %399, 121
  %444 = mul i64 %403, 11
  %445 = add i64 %443, %444
  %446 = add i64 %445, %407
  %447 = getelementptr i64, ptr %71, i64 %446
  %448 = load i64, ptr %447, align 4
  %449 = call i64 @__float_add(i64 %442, i64 %448, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1)
  %450 = mul i64 %399, 121
  %451 = mul i64 %403, 11
  %452 = add i64 %450, %451
  %453 = add i64 %452, %407
  %454 = getelementptr i64, ptr %71, i64 %453
  store i64 %449, ptr %454, align 4
  %455 = add i64 %411, 1
  br label %410

456:                                              ; preds = %410
  br label %457

457:                                              ; preds = %460, %456
  %458 = phi i64 [ %487, %460 ], [ 10, %456 ]
  %459 = icmp slt i64 %458, 21
  br i1 %459, label %460, label %488

460:                                              ; preds = %457
  %461 = add i64 %458, -10
  %462 = mul i64 %407, 11
  %463 = add i64 %462, %461
  %464 = getelementptr i64, ptr %1, i64 %463
  %465 = load i64, ptr %464, align 4
  %466 = mul i64 %399, 121
  %467 = mul i64 %403, 11
  %468 = add i64 %466, %467
  %469 = add i64 %468, %407
  %470 = getelementptr i64, ptr %71, i64 %469
  %471 = load i64, ptr %470, align 4
  %472 = call i64 @__float_mul(i64 %465, i64 %471, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1)
  %473 = add i64 %458, -10
  %474 = mul i64 %473, 121
  %475 = mul i64 %399, 11
  %476 = add i64 %474, %475
  %477 = add i64 %476, %403
  %478 = getelementptr i64, ptr %26, i64 %477
  %479 = load i64, ptr %478, align 4
  %480 = call i64 @__float_add(i64 %472, i64 %479, i8 11, i8 52, i32 -1023, i1 true, i1 true, i1 true, i1 true, i8 -1)
  %481 = add i64 %458, -10
  %482 = mul i64 %481, 121
  %483 = mul i64 %399, 11
  %484 = add i64 %482, %483
  %485 = add i64 %484, %403
  %486 = getelementptr i64, ptr %26, i64 %485
  store i64 %480, ptr %486, align 4
  %487 = add i64 %458, 1
  br label %457

488:                                              ; preds = %457
  %489 = add i64 %407, 1
  br label %406

490:                                              ; preds = %406
  %491 = add i64 %403, 1
  br label %402

492:                                              ; preds = %402
  %493 = add i64 %399, 1
  br label %398

494:                                              ; preds = %398
  ret void
}

define void @_mlir_ciface_kernel_sf(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9) {
  %11 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %0, align 8
  %12 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 0
  %13 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 1
  %14 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 2
  %15 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 3, 0
  %16 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 3, 1
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 4, 0
  %18 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, 4, 1
  %19 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %1, align 8
  %20 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %19, 0
  %21 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %19, 1
  %22 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %19, 2
  %23 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %19, 3, 0
  %24 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %19, 3, 1
  %25 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %19, 3, 2
  %26 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %19, 4, 0
  %27 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %19, 4, 1
  %28 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %19, 4, 2
  %29 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %2, align 8
  %30 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, 0
  %31 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, 1
  %32 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, 2
  %33 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, 3, 0
  %34 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, 3, 1
  %35 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, 3, 2
  %36 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, 4, 0
  %37 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, 4, 1
  %38 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, 4, 2
  %39 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %3, align 8
  %40 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %39, 0
  %41 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %39, 1
  %42 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %39, 2
  %43 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %39, 3, 0
  %44 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %39, 3, 1
  %45 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %39, 3, 2
  %46 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %39, 4, 0
  %47 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %39, 4, 1
  %48 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %39, 4, 2
  %49 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %4, align 8
  %50 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %49, 0
  %51 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %49, 1
  %52 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %49, 2
  %53 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %49, 3, 0
  %54 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %49, 3, 1
  %55 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %49, 3, 2
  %56 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %49, 4, 0
  %57 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %49, 4, 1
  %58 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %49, 4, 2
  %59 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %5, align 8
  %60 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %59, 0
  %61 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %59, 1
  %62 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %59, 2
  %63 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %59, 3, 0
  %64 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %59, 3, 1
  %65 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %59, 3, 2
  %66 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %59, 4, 0
  %67 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %59, 4, 1
  %68 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %59, 4, 2
  %69 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %6, align 8
  %70 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %69, 0
  %71 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %69, 1
  %72 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %69, 2
  %73 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %69, 3, 0
  %74 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %69, 3, 1
  %75 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %69, 3, 2
  %76 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %69, 4, 0
  %77 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %69, 4, 1
  %78 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %69, 4, 2
  %79 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %7, align 8
  %80 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %79, 0
  %81 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %79, 1
  %82 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %79, 2
  %83 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %79, 3, 0
  %84 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %79, 3, 1
  %85 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %79, 3, 2
  %86 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %79, 4, 0
  %87 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %79, 4, 1
  %88 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %79, 4, 2
  %89 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %8, align 8
  %90 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %89, 0
  %91 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %89, 1
  %92 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %89, 2
  %93 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %89, 3, 0
  %94 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %89, 3, 1
  %95 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %89, 3, 2
  %96 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %89, 4, 0
  %97 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %89, 4, 1
  %98 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %89, 4, 2
  %99 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %9, align 8
  %100 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %99, 0
  %101 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %99, 1
  %102 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %99, 2
  %103 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %99, 3, 0
  %104 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %99, 3, 1
  %105 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %99, 3, 2
  %106 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %99, 4, 0
  %107 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %99, 4, 1
  %108 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %99, 4, 2
  call void @kernel_sf(ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, ptr %30, ptr %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36, i64 %37, i64 %38, ptr %40, ptr %41, i64 %42, i64 %43, i64 %44, i64 %45, i64 %46, i64 %47, i64 %48, ptr %50, ptr %51, i64 %52, i64 %53, i64 %54, i64 %55, i64 %56, i64 %57, i64 %58, ptr %60, ptr %61, i64 %62, i64 %63, i64 %64, i64 %65, i64 %66, i64 %67, i64 %68, ptr %70, ptr %71, i64 %72, i64 %73, i64 %74, i64 %75, i64 %76, i64 %77, i64 %78, ptr %80, ptr %81, i64 %82, i64 %83, i64 %84, i64 %85, i64 %86, i64 %87, i64 %88, ptr %90, ptr %91, i64 %92, i64 %93, i64 %94, i64 %95, i64 %96, i64 %97, i64 %98, ptr %100, ptr %101, i64 %102, i64 %103, i64 %104, i64 %105, i64 %106, i64 %107, i64 %108)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
