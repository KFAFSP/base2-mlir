#ifndef BASE2_FIXEDOPS
#define BASE2_FIXEDOPS

// Template for fixed-point ops.
class Base2_FixedPointOpBase<string mnemonic, list<Trait> traits = []>
        : Base2_Op<
            "fixed." # mnemonic,
            traits # [
                Pure,
                SameOperandsAndResultShape
            ] # ElementwiseMappable.traits> {
    let arguments = (ins
        Base2_FixedPointLike:$lhs,
        Base2_FixedPointLike:$rhs);
    let results = (outs Base2_FixedPointLike:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($result)
    }];
}

def Base2_FixedAddOp
        : Base2_FixedPointOpBase<
            "add",
            [
                Commutative,
                DeclareOpInterfaceMethods<
                    InferTypeOpInterface,
                    ["inferReturnTypes"]>]> {
    let summary = "Exact addition operation";
    let description = [{
        The `fixed.add` operation computes the sum of two FixedPointType numbers.
        If the value is Signless, an error will be returned, because this is the
        exact operation between numbers. The result type will be solved using
        `inferReturnTypes` method.

        Syntax:

        ```
        fixed-add-op ::= `fixed.add` $lhs, $rhs `:` type($lhs) `,` type($rhs)
        ```

        Example:

        ```
        %0 = base2.fixed.add %1, %2 : !base2.si4_4, !base2.ui4_4
        return %0 : !base2.si6_4
        ```
    }];

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
    }];
}


#endif
