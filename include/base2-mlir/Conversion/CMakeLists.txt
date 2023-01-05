set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Base2Conversion)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Base2Conversion)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Base2Conversion)
add_public_tablegen_target(Base2ConversionPassIncGen)

add_mlir_doc(Passes ConversionPasses ./ -gen-pass-doc)