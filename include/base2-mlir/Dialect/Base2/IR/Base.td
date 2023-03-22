//===- Base.td ---------------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Base2 dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef BASE2_BASE
#define BASE2_BASE

include "base2-mlir/Dialect/Base2/Enums.td"
include "base2-mlir/Dialect/Base2/Interfaces/Interfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def Base2_Dialect : Dialect {
    let name = "base2";
    let cppNamespace = "::mlir::base2";

    let summary = "The binary number representation dialect";
    let description = [{
        A dialect that defines and implements concise semantics for types and
        operations pertaining to numbers represented in the most ubiquitous
        binary formats.
    }];

    let hasConstantMaterializer = 1;
    let useDefaultTypePrinterParser = 0;
    let useFoldAPI = kEmitFoldAdaptorFolder;

    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();

        Type parseType(DialectAsmParser &parser) const override;
        void printType(Type type, DialectAsmPrinter &printer) const override;
    }];
}

// Template for ops.
class Base2_Op<string mnemonic, list<Trait> traits = []>
        : Op<Base2_Dialect, mnemonic, traits>;
// Template for types.
class Base2_Type<string name, list<Trait> traits = []>
        : TypeDef<Base2_Dialect, name, traits>;

#endif
