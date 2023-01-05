# SoftFloat Dialect

- [SoftFloat Dialect](#softfloat-dialect)
  - [`SoftFloat Library`](#softfloat-library)
  - [`SoftFloat Operations`](#softfloat-operations)
  - [`SoftFloat Lowering`](#softfloat-lowering)
  - [`Future Works?`](#future-works)

## `SoftFloat Library`

[SoftFloat](http://www.jhauser.us/arithmetic/SoftFloat.html) library is a software implementation of binary floating-point that conforms to the [IEEE754](https://standards.ieee.org/ieee/754/6210/) Standard for Floating-Point Arithmetic by Berkeley.

The version `SoftFloat` dialect uses is a custom version implementation with in the [Bambu](https://github.com/ferrandi/PandA-bambu/tree/5e5e306b86383a7d85274d64977a3d71fdcff4fe/etc/libbambu/softfloat) project, in which `64-bit` numbers are accepted. That means, this `SoftFloat` supports the operations on `double precision` floating point number. For this we inroduce a type `softfloat.sfloat` into this dialect to make everybody who wants to use this dialect not able to use any other types. At last it will be lowered to the `i64` type.

Arbitrary preicision is also supported in this library by passing the semantics to the operations, which will be lowering into `func.call`.

## `SoftFloat Operations`

Right now `SoftFloat` dialect supports following operations:

- `Arithmetic Operations`
  - `Addition(softfloat.add)`
  - `Substraction(softfloat.sub)`
  - `Multiplication(softfloat.mul)`
  - `Division`
    - `Goldschmidt Algorithm(softfloat.divg)`
    - `SRT4 Algorithm(softfloat.divsrt)`
- `Comparison Operations`
  - `Equal(softfloat.add)`
  - `Less than(softfloat.add)`
  - `Less than or equal(softfloat.add)`
  - `Greater than(softfloat.add)`
  - `Greater than or equal(softfloat.add)`
  - `Not equal(softfloat.ltgt)`
  - `Is a signaling NaN(softfloat.nan)`
- `Cast Operations`
  - `Cast between two sfloat numbers(softfloat.cast)`
  - `Cast a f64 number into sfloat number(softfloat.castfloat)`
  - `Cast a sfloat number into f64 number(softfloat.casttofloat)`

The operands we need to feed to the operations are `lhs`, `rhs`, `exp_bits`, `frac_bits`, `exp_bias`, `has_rounding`, `has_nan`, `has_one`, `has_subnorm`, `sign`. One of the sad thing is that this library only support `truncation` rounding mode of `IEEE754 Standard`. For `softfloat.cast` we need to give both the semantics for `lhs(from)` and `rhs(to)`. `sign` can be set to `-1` to let the library automatically compute the signedness of the result.

## `SoftFloat Lowering`

The `SoftFloat` dialect is 1:1 maped to the library, for which the lowering pass will replace the operation with a `func.call` to the actual function in the library. During the lowering the `sfloat` type will be converted into `i64` type using `typeConverter`. As for the `softfloat.castfloat/casttofloat` we use the `arith.bitcast` to do a bitcast between `f64` and `i64`, which is `sfloat`.

We also need to rewrite `affine.load/store` operations to let the read/write from/to a `memref` type of `sfloat` will be converted into `memref` of `i64`.

## `Future Works?`

- Support for lower or bigger bit-width number by making `SoftFloat` parametric.
- Rewrite `linalg.generic` operations