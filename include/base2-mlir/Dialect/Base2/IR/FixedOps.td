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

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
    }];
}


#endif
