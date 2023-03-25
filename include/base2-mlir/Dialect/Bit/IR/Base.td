//===- Base.td ---------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Bit dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef BIT_BASE
#define BIT_BASE

include "base2-mlir/Dialect/Bit/Enums.td"
include "base2-mlir/Dialect/Bit/Interfaces/Interfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def Bit_Dialect : Dialect {
    let name = "bit";
    let cppNamespace = "::mlir::bit";

    let summary = "The bit sequence manipulation dialect";
    let description = [{
        A dialect that defines the concept of ordered bit sequences and provides
        attributes and operations for declaring, examining and computing them.
    }];

    let dependentDialects = [
        "index::IndexDialect"
    ];

    let hasConstantMaterializer = 1;
    let useDefaultAttributePrinterParser = 1;
    let useFoldAPI = kEmitFoldAdaptorFolder;

    code extraClassDeclaration = [{
    private:
        void registerAttributes();
        void registerOps();
    }];
}

// Template for attributes.
class Bit_Attr<string name, list<Trait> traits = []>
        : AttrDef<Bit_Dialect, name, traits>;
// Template for ops.
class Bit_Op<string mnemonic, list<Trait> traits = []>
        : Op<Bit_Dialect, mnemonic, traits>;

#endif
