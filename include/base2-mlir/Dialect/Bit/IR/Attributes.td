//===- Attributes.td ---------------------------------------*- tablegen -*-===//
//
// This is the definitions file for the Bit dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef BIT_ATTRIBUTES
#define BIT_ATTRIBUTES

include "base2-mlir/Dialect/Bit/IR/Base.td"

//===----------------------------------------------------------------------===//
// BitsAttr
//===----------------------------------------------------------------------===//

def Bit_BitsAttr : Bit_Attr<"Bits", [TypedAttrInterface, Bit_BitSequenceAttr]> {
    let mnemonic = "bits";
    let summary = "Constant bit sequence attribute";
    let description = [{
        The `bits` attribute represents a compile-time constant sequence of bits
        that defines the value of a `BitSequenceType`.

        Syntax:

        ```
        bits-attr    ::= `bits` `<` bit-sequence `>` `:` type
        bit-sequence ::= `"` bit-literal (trunc-spec)? `"`
        bit-literal  ::= `0b` [_01]* | `0x` [_0-9a-fA-F]*
        trunc-spec   ::= `/` [0-9]+
        ```

        The number of bits in the bit-sequence must be equal to the bit width
        of the type.

        Example:

        ```
        // These all have the same value.
        #bit.bits<"0b0000001000"> : i10
        #bit.bits<"0b1000/10"> : i10
        #bit.bits<"0x8/10"> : i10
        #bit.bits<"0x0008/10"> : i10
        #bit.bits<"0x000008/10"> : i10

        // And this is the canonical version of the same value.
        #bit.bits<"0x08/10"> : i10
        ```
    }];

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        "BitSequence":$value);

    let builders = [
        AttrBuilderWithInferredContext<(ins
            "BitSequenceType":$type,
            "const BitSequence &":$value),
        [{
            return $_get(type.getContext(), type, value);
        }]>];
    let skipDefaultBuilders = 1;

    let assemblyFormat = "`<`$value`>`";

    let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// DenseBitsAttr
//===----------------------------------------------------------------------===//

// NOTE: We cannot implement TypedAttrInterface, because then the printer would
//       repeat the stored element type. We can't use a trailing type, because
//       we must parse the data with a known type.

def Bit_DenseBitsAttr : Bit_Attr<"DenseBits", [ElementsAttrInterface]> {
    let mnemonic = "dense_bits";
    let summary = "Constant densely-packed bit sequence container attribute";
    let description = [{
        The `dense_bits` attribute represents a compile-time constant container
        of sequences of bits that each define the value of a common
        `BitSequenceType` element type.

        Syntax:

        ```
        dense-bits-attr ::= `dense_bits` `<` $type `=` $data `>`
        ```

        The result type must be a shaped type of `BitSequenceType` elements,
        and the dense elements contained must match the shape of type, or be a
        splat.

        Example:

        ```
        // These all have the same value.
        #bit.dense_bits<tensor<3xi10> = dense<1>>
        #bit.dense_bits<tensor<3xi10> = dense<[1, 2, 3]>>
        #bit.dense_bits<tensor<3xi10> = dense<"0x010002000300">>
        ```
    }];

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        "DenseIntElementsAttr":$data);

    let builders = [
        AttrBuilderWithInferredContext<(ins
            "BitSequenceType":$type,
            "const DenseIntElementsAttr":$data),
        [{
            return $_get(type.getContext(), type, data);
        }]>];
    let skipDefaultBuilders = 1;

    let hasCustomAssemblyFormat = 1;

    let genVerifyDecl = 1;

    code extraClassDeclaration = [{
    public:
        //===--------------------------------------------------------------===//
        // ElementsAttrInterface
        //===--------------------------------------------------------------===//

        using ContiguousIterableTypesT =
            std::tuple<std::uint8_t, std::uint64_t>;

        auto try_value_begin_impl(OverloadToken<std::uint8_t>) const
        {
            return getData().try_value_begin<std::uint8_t>();
        }
        auto try_value_begin_impl(OverloadToken<std::uint64_t>) const
        {
            return getData().try_value_begin<std::uint64_t>();
        }

        using NonContiguousIterableTypesT = std::tuple<BitSequence>;

        auto try_value_begin_impl(OverloadToken<BitSequence>) const
        {
            const auto bitWidth = getElementType()
                .cast<BitSequenceType>()
                .getBitWidth();
            return mlir::success(
                llvm::map_range(
                    getData().getValues<llvm::APInt>(),
                    [=](const llvm::APInt &raw) {
                        return BitSequence(raw).getLSB(bitWidth);
                    })
                    .begin());
        }
    }];
}

#endif
