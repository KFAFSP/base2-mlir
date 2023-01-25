//===- Base.td - Base2 dialect base ------------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef SOFTFLOAT_BASE
#define SOFTFLOAT_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def SoftFloat_Dialect : Dialect {
    let name = "softfloat";
    let cppNamespace = "::mlir::softfloat";

    let summary = "A software implementation for floating point";
    let description = [{
        SoftFloat is a software implementation of floating-point
        that conforms to the IEC/IEEE Standard for Binary Floating-Point
        Arithmetic
    }];

    // let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
    let useDefaultTypePrinterParser = 1;

    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
    }];
}

// Template for attributes.
// class SoftFloat_Attr<string name, list<Trait> traits = []>
//         : AttrDef<SoftFloat_Dialect, name, traits>;
// Template for ops.
class SoftFloat_Op<string mnemonic, list<Trait> traits = []>
        : Op<SoftFloat_Dialect, mnemonic, traits>;
// Template for types.
class SoftFloat_Type<string name, list<Trait> traits = []>
        : TypeDef<SoftFloat_Dialect, name, traits>;

#endif
