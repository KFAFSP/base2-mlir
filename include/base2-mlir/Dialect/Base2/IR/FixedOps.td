#ifndef BASE2_FIXEDOPS
#define BASE2_FIXEDOPS

// Template for fixed-point ops.
class Base2_FixedPointOpBase<string mnemonic, list<Trait> traits = []>
        : Base2_Op<
            "fixed." # mnemonic,
            traits # [
                Pure,
            ] # ElementwiseMappable.traits> {
    let arguments = (ins
        Base2_FixedPointLike:$lhs,
        Base2_FixedPointLike:$rhs);
    let results = (outs Base2_FixedPointLike:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($result)
    }];

    let hasFolder = 1;
}

def Base2_FixedPromoteOp
        : Base2_FixedPointOpBase<"promote", [OpAsmOpInterface, Commutative]> {
    let summary = "Fixed-point promotion";
    let description = [{
        The `promote` operation value casts one or more fixed-point operands to
        a common supertype. The values will always be exactly representable.

        Syntax:

        ```
        promote-op  ::= `fixed.promote` operands `:` type(operands)
                            `to` $resultType
        ```

        Example:

        ```
        %2, %3 = fixed.promote %0, %1 : !base2.si7_3, !base2.ui9_1
                                        to !base2.si9_3
        ```
    }];

    let arguments = (ins Variadic<Type<Base2_FixedPointLike.predicate>>);
    let results = (outs Variadic<Type<Base2_FixedPointLike.predicate>>);

    let builders = [
        OpBuilder<(ins "Value":$value, "FixedPointSemantics":$resultTy)>,
        OpBuilder<(ins
            "ValueRange":$values,
            CArg<"FixedPointSemantics", "{}">:$resultTy)>
    ];

    let assemblyFormat = ?;
    let hasCustomAssemblyFormat = 1;

    let hasVerifier = 1;

    code extraClassDeclaration = [{
    public:
        //===--------------------------------------------------------------===//
        // OpAsmOpInterface
        //===--------------------------------------------------------------===//

        void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
        {
            for (auto res : getResults()) setNameFn(res, "ext");
        }
    }];
}

#endif
