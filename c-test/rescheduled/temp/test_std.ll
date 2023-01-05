; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define void @kernel_std(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24, ptr %25, ptr %26, i64 %27, i64 %28, i64 %29, i64 %30, i64 %31, i64 %32, i64 %33, ptr %34, ptr %35, i64 %36, i64 %37, i64 %38, i64 %39, i64 %40, i64 %41, i64 %42, ptr %43, ptr %44, i64 %45, i64 %46, i64 %47, i64 %48, i64 %49, i64 %50, i64 %51, ptr %52, ptr %53, i64 %54, i64 %55, i64 %56, i64 %57, i64 %58, i64 %59, i64 %60, ptr %61, ptr %62, i64 %63, i64 %64, i64 %65, i64 %66, i64 %67, i64 %68, i64 %69, ptr %70, ptr %71, i64 %72, i64 %73, i64 %74, i64 %75, i64 %76, i64 %77, i64 %78, ptr %79, ptr %80, i64 %81, i64 %82, i64 %83, i64 %84, i64 %85, i64 %86, i64 %87) {
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
  br label %177

177:                                              ; preds = %273, %88
  %178 = phi i64 [ %274, %273 ], [ 0, %88 ]
  %179 = icmp slt i64 %178, 11
  br i1 %179, label %180, label %275

180:                                              ; preds = %177
  br label %181

181:                                              ; preds = %271, %180
  %182 = phi i64 [ %272, %271 ], [ 0, %180 ]
  %183 = icmp slt i64 %182, 11
  br i1 %183, label %184, label %273

184:                                              ; preds = %181
  br label %185

185:                                              ; preds = %269, %184
  %186 = phi i64 [ %270, %269 ], [ 0, %184 ]
  %187 = icmp slt i64 %186, 11
  br i1 %187, label %188, label %271

188:                                              ; preds = %185
  %189 = mul i64 %178, 121
  %190 = mul i64 %182, 11
  %191 = add i64 %189, %190
  %192 = add i64 %191, %186
  %193 = getelementptr double, ptr %62, i64 %192
  store double 0.000000e+00, ptr %193, align 8
  br label %194

194:                                              ; preds = %267, %188
  %195 = phi i64 [ %268, %267 ], [ 0, %188 ]
  %196 = icmp slt i64 %195, 21
  br i1 %196, label %197, label %269

197:                                              ; preds = %194
  %198 = mul i64 %195, -1
  %199 = add i64 %198, 10
  %200 = icmp sge i64 %199, 0
  br i1 %200, label %201, label %225

201:                                              ; preds = %197
  %202 = mul i64 %178, 11
  %203 = add i64 %202, %195
  %204 = getelementptr double, ptr %1, i64 %203
  %205 = load double, ptr %204, align 8
  %206 = mul i64 %182, 121
  %207 = mul i64 %186, 11
  %208 = add i64 %206, %207
  %209 = add i64 %208, %195
  %210 = getelementptr double, ptr %17, i64 %209
  %211 = load double, ptr %210, align 8
  %212 = fmul double %205, %211
  %213 = mul i64 %178, 121
  %214 = mul i64 %182, 11
  %215 = add i64 %213, %214
  %216 = add i64 %215, %186
  %217 = getelementptr double, ptr %62, i64 %216
  %218 = load double, ptr %217, align 8
  %219 = fadd double %212, %218
  %220 = mul i64 %178, 121
  %221 = mul i64 %182, 11
  %222 = add i64 %220, %221
  %223 = add i64 %222, %186
  %224 = getelementptr double, ptr %62, i64 %223
  store double %219, ptr %224, align 8
  br label %225

225:                                              ; preds = %201, %197
  %226 = add i64 %195, -10
  %227 = icmp sge i64 %226, 0
  br i1 %227, label %228, label %255

228:                                              ; preds = %225
  %229 = add i64 %195, -10
  %230 = mul i64 %186, 11
  %231 = add i64 %230, %229
  %232 = getelementptr double, ptr %1, i64 %231
  %233 = load double, ptr %232, align 8
  %234 = mul i64 %178, 121
  %235 = mul i64 %182, 11
  %236 = add i64 %234, %235
  %237 = add i64 %236, %186
  %238 = getelementptr double, ptr %62, i64 %237
  %239 = load double, ptr %238, align 8
  %240 = fmul double %233, %239
  %241 = add i64 %195, -10
  %242 = mul i64 %241, 121
  %243 = mul i64 %178, 11
  %244 = add i64 %242, %243
  %245 = add i64 %244, %182
  %246 = getelementptr double, ptr %53, i64 %245
  %247 = load double, ptr %246, align 8
  %248 = fadd double %240, %247
  %249 = add i64 %195, -10
  %250 = mul i64 %249, 121
  %251 = mul i64 %178, 11
  %252 = add i64 %250, %251
  %253 = add i64 %252, %182
  %254 = getelementptr double, ptr %53, i64 %253
  store double %248, ptr %254, align 8
  br label %255

255:                                              ; preds = %228, %225
  %256 = icmp eq i64 %186, 0
  %257 = mul i64 %195, -1
  %258 = add i64 %257, 10
  %259 = icmp sge i64 %258, 0
  %260 = and i1 %256, %259
  br i1 %260, label %261, label %267

261:                                              ; preds = %255
  %262 = mul i64 %195, 121
  %263 = mul i64 %178, 11
  %264 = add i64 %262, %263
  %265 = add i64 %264, %182
  %266 = getelementptr double, ptr %53, i64 %265
  store double 0.000000e+00, ptr %266, align 8
  br label %267

267:                                              ; preds = %261, %255
  %268 = add i64 %195, 1
  br label %194

269:                                              ; preds = %194
  %270 = add i64 %186, 1
  br label %185

271:                                              ; preds = %185
  %272 = add i64 %182, 1
  br label %181

273:                                              ; preds = %181
  %274 = add i64 %178, 1
  br label %177

275:                                              ; preds = %177
  br label %276

276:                                              ; preds = %394, %275
  %277 = phi i64 [ %395, %394 ], [ 0, %275 ]
  %278 = icmp slt i64 %277, 11
  br i1 %278, label %279, label %396

279:                                              ; preds = %276
  br label %280

280:                                              ; preds = %392, %279
  %281 = phi i64 [ %393, %392 ], [ 0, %279 ]
  %282 = icmp slt i64 %281, 11
  br i1 %282, label %283, label %394

283:                                              ; preds = %280
  br label %284

284:                                              ; preds = %390, %283
  %285 = phi i64 [ %391, %390 ], [ 0, %283 ]
  %286 = icmp slt i64 %285, 11
  br i1 %286, label %287, label %392

287:                                              ; preds = %284
  %288 = mul i64 %277, 121
  %289 = mul i64 %281, 11
  %290 = add i64 %288, %289
  %291 = add i64 %290, %285
  %292 = getelementptr double, ptr %35, i64 %291
  store double 0.000000e+00, ptr %292, align 8
  br label %293

293:                                              ; preds = %388, %287
  %294 = phi i64 [ %389, %388 ], [ 0, %287 ]
  %295 = icmp slt i64 %294, 21
  br i1 %295, label %296, label %390

296:                                              ; preds = %293
  %297 = mul i64 %294, -1
  %298 = add i64 %297, 10
  %299 = icmp sge i64 %298, 0
  br i1 %299, label %300, label %346

300:                                              ; preds = %296
  %301 = mul i64 %277, 11
  %302 = add i64 %301, %294
  %303 = getelementptr double, ptr %1, i64 %302
  %304 = load double, ptr %303, align 8
  %305 = mul i64 %281, 121
  %306 = mul i64 %285, 11
  %307 = add i64 %305, %306
  %308 = add i64 %307, %294
  %309 = getelementptr double, ptr %53, i64 %308
  %310 = load double, ptr %309, align 8
  %311 = fmul double %304, %310
  %312 = mul i64 %277, 121
  %313 = mul i64 %281, 11
  %314 = add i64 %312, %313
  %315 = add i64 %314, %285
  %316 = getelementptr double, ptr %35, i64 %315
  %317 = load double, ptr %316, align 8
  %318 = fadd double %311, %317
  %319 = mul i64 %277, 121
  %320 = mul i64 %281, 11
  %321 = add i64 %319, %320
  %322 = add i64 %321, %285
  %323 = getelementptr double, ptr %35, i64 %322
  store double %318, ptr %323, align 8
  %324 = add i64 %294, -10
  %325 = icmp eq i64 %324, 0
  br i1 %325, label %326, label %345

326:                                              ; preds = %300
  %327 = mul i64 %277, 121
  %328 = mul i64 %281, 11
  %329 = add i64 %327, %328
  %330 = add i64 %329, %285
  %331 = getelementptr double, ptr %8, i64 %330
  %332 = load double, ptr %331, align 8
  %333 = mul i64 %277, 121
  %334 = mul i64 %281, 11
  %335 = add i64 %333, %334
  %336 = add i64 %335, %285
  %337 = getelementptr double, ptr %35, i64 %336
  %338 = load double, ptr %337, align 8
  %339 = fmul double %332, %338
  %340 = mul i64 %277, 121
  %341 = mul i64 %281, 11
  %342 = add i64 %340, %341
  %343 = add i64 %342, %285
  %344 = getelementptr double, ptr %44, i64 %343
  store double %339, ptr %344, align 8
  br label %345

345:                                              ; preds = %326, %300
  br label %346

346:                                              ; preds = %345, %296
  %347 = add i64 %294, -10
  %348 = icmp sge i64 %347, 0
  br i1 %348, label %349, label %376

349:                                              ; preds = %346
  %350 = add i64 %294, -10
  %351 = mul i64 %285, 11
  %352 = add i64 %351, %350
  %353 = getelementptr double, ptr %1, i64 %352
  %354 = load double, ptr %353, align 8
  %355 = mul i64 %277, 121
  %356 = mul i64 %281, 11
  %357 = add i64 %355, %356
  %358 = add i64 %357, %285
  %359 = getelementptr double, ptr %35, i64 %358
  %360 = load double, ptr %359, align 8
  %361 = fmul double %354, %360
  %362 = add i64 %294, -10
  %363 = mul i64 %362, 121
  %364 = mul i64 %277, 11
  %365 = add i64 %363, %364
  %366 = add i64 %365, %281
  %367 = getelementptr double, ptr %80, i64 %366
  %368 = load double, ptr %367, align 8
  %369 = fadd double %361, %368
  %370 = add i64 %294, -10
  %371 = mul i64 %370, 121
  %372 = mul i64 %277, 11
  %373 = add i64 %371, %372
  %374 = add i64 %373, %281
  %375 = getelementptr double, ptr %80, i64 %374
  store double %369, ptr %375, align 8
  br label %376

376:                                              ; preds = %349, %346
  %377 = icmp eq i64 %285, 0
  %378 = mul i64 %294, -1
  %379 = add i64 %378, 10
  %380 = icmp sge i64 %379, 0
  %381 = and i1 %377, %380
  br i1 %381, label %382, label %388

382:                                              ; preds = %376
  %383 = mul i64 %294, 121
  %384 = mul i64 %277, 11
  %385 = add i64 %383, %384
  %386 = add i64 %385, %281
  %387 = getelementptr double, ptr %80, i64 %386
  store double 0.000000e+00, ptr %387, align 8
  br label %388

388:                                              ; preds = %382, %376
  %389 = add i64 %294, 1
  br label %293

390:                                              ; preds = %293
  %391 = add i64 %285, 1
  br label %284

392:                                              ; preds = %284
  %393 = add i64 %281, 1
  br label %280

394:                                              ; preds = %280
  %395 = add i64 %277, 1
  br label %276

396:                                              ; preds = %276
  br label %397

397:                                              ; preds = %491, %396
  %398 = phi i64 [ %492, %491 ], [ 0, %396 ]
  %399 = icmp slt i64 %398, 11
  br i1 %399, label %400, label %493

400:                                              ; preds = %397
  br label %401

401:                                              ; preds = %489, %400
  %402 = phi i64 [ %490, %489 ], [ 0, %400 ]
  %403 = icmp slt i64 %402, 11
  br i1 %403, label %404, label %491

404:                                              ; preds = %401
  br label %405

405:                                              ; preds = %487, %404
  %406 = phi i64 [ %488, %487 ], [ 0, %404 ]
  %407 = icmp slt i64 %406, 11
  br i1 %407, label %408, label %489

408:                                              ; preds = %405
  br label %409

409:                                              ; preds = %430, %408
  %410 = phi i64 [ %454, %430 ], [ 0, %408 ]
  %411 = icmp slt i64 %410, 11
  br i1 %411, label %412, label %455

412:                                              ; preds = %409
  %413 = add i64 %406, -10
  %414 = icmp eq i64 %413, 0
  br i1 %414, label %415, label %421

415:                                              ; preds = %412
  %416 = mul i64 %410, 121
  %417 = mul i64 %398, 11
  %418 = add i64 %416, %417
  %419 = add i64 %418, %402
  %420 = getelementptr double, ptr %26, i64 %419
  store double 0.000000e+00, ptr %420, align 8
  br label %421

421:                                              ; preds = %415, %412
  %422 = add i64 %410, -10
  %423 = icmp eq i64 %422, 0
  br i1 %423, label %424, label %430

424:                                              ; preds = %421
  %425 = mul i64 %398, 121
  %426 = mul i64 %402, 11
  %427 = add i64 %425, %426
  %428 = add i64 %427, %406
  %429 = getelementptr double, ptr %71, i64 %428
  store double 0.000000e+00, ptr %429, align 8
  br label %430

430:                                              ; preds = %424, %421
  %431 = mul i64 %410, 11
  %432 = add i64 %431, %398
  %433 = getelementptr double, ptr %1, i64 %432
  %434 = load double, ptr %433, align 8
  %435 = mul i64 %402, 121
  %436 = mul i64 %406, 11
  %437 = add i64 %435, %436
  %438 = add i64 %437, %410
  %439 = getelementptr double, ptr %80, i64 %438
  %440 = load double, ptr %439, align 8
  %441 = fmul double %434, %440
  %442 = mul i64 %398, 121
  %443 = mul i64 %402, 11
  %444 = add i64 %442, %443
  %445 = add i64 %444, %406
  %446 = getelementptr double, ptr %71, i64 %445
  %447 = load double, ptr %446, align 8
  %448 = fadd double %441, %447
  %449 = mul i64 %398, 121
  %450 = mul i64 %402, 11
  %451 = add i64 %449, %450
  %452 = add i64 %451, %406
  %453 = getelementptr double, ptr %71, i64 %452
  store double %448, ptr %453, align 8
  %454 = add i64 %410, 1
  br label %409

455:                                              ; preds = %409
  br label %456

456:                                              ; preds = %459, %455
  %457 = phi i64 [ %486, %459 ], [ 10, %455 ]
  %458 = icmp slt i64 %457, 21
  br i1 %458, label %459, label %487

459:                                              ; preds = %456
  %460 = add i64 %457, -10
  %461 = mul i64 %406, 11
  %462 = add i64 %461, %460
  %463 = getelementptr double, ptr %1, i64 %462
  %464 = load double, ptr %463, align 8
  %465 = mul i64 %398, 121
  %466 = mul i64 %402, 11
  %467 = add i64 %465, %466
  %468 = add i64 %467, %406
  %469 = getelementptr double, ptr %71, i64 %468
  %470 = load double, ptr %469, align 8
  %471 = fmul double %464, %470
  %472 = add i64 %457, -10
  %473 = mul i64 %472, 121
  %474 = mul i64 %398, 11
  %475 = add i64 %473, %474
  %476 = add i64 %475, %402
  %477 = getelementptr double, ptr %26, i64 %476
  %478 = load double, ptr %477, align 8
  %479 = fadd double %471, %478
  %480 = add i64 %457, -10
  %481 = mul i64 %480, 121
  %482 = mul i64 %398, 11
  %483 = add i64 %481, %482
  %484 = add i64 %483, %402
  %485 = getelementptr double, ptr %26, i64 %484
  store double %479, ptr %485, align 8
  %486 = add i64 %457, 1
  br label %456

487:                                              ; preds = %456
  %488 = add i64 %406, 1
  br label %405

489:                                              ; preds = %405
  %490 = add i64 %402, 1
  br label %401

491:                                              ; preds = %401
  %492 = add i64 %398, 1
  br label %397

493:                                              ; preds = %397
  ret void
}

define void @_mlir_ciface_kernel_std(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9) {
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
  call void @kernel_std(ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, i64 %17, i64 %18, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, i64 %28, ptr %30, ptr %31, i64 %32, i64 %33, i64 %34, i64 %35, i64 %36, i64 %37, i64 %38, ptr %40, ptr %41, i64 %42, i64 %43, i64 %44, i64 %45, i64 %46, i64 %47, i64 %48, ptr %50, ptr %51, i64 %52, i64 %53, i64 %54, i64 %55, i64 %56, i64 %57, i64 %58, ptr %60, ptr %61, i64 %62, i64 %63, i64 %64, i64 %65, i64 %66, i64 %67, i64 %68, ptr %70, ptr %71, i64 %72, i64 %73, i64 %74, i64 %75, i64 %76, i64 %77, i64 %78, ptr %80, ptr %81, i64 %82, i64 %83, i64 %84, i64 %85, i64 %86, i64 %87, i64 %88, ptr %90, ptr %91, i64 %92, i64 %93, i64 %94, i64 %95, i64 %96, i64 %97, i64 %98, ptr %100, ptr %101, i64 %102, i64 %103, i64 %104, i64 %105, i64 %106, i64 %107, i64 %108)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
