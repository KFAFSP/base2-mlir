# Abstract

The Base2 dialect aims to provide a comprehensive abstraction for number types based on binary representations and operations on them.

- [Abstract](#abstract)
  - [Bit sequences](#bit-sequences)
    - [`BitSequence`](#bitsequence)
    - [`BitSequenceType`](#bitsequencetype)
    - [`BitSequenceLikeType`](#bitsequenceliketype)
    - [`BitSequenceAttr`](#bitsequenceattr)
      - [Canonicalization](#canonicalization)
    - [`BitSequenceLikeAttr`](#bitsequencelikeattr)
      - [Canonicalization](#canonicalization-1)
    - [`base2.constant`](#base2constant)
    - [`base2.bit_cast`](#base2bit_cast)
      - [Folding](#folding)
  - [Binary numerals](#binary-numerals)
    - [`InterpretableType`](#interpretabletype)
    - [`InterpretableLikeType`](#interpretableliketype)
    - [`base2.value_cast`](#base2value_cast)
      - [Folding](#folding-1)
    - [Common operations](#common-operations)
    - [Closed arithmetic](#closed-arithmetic)
  - [Fixed-point numbers](#fixed-point-numbers)
  - [Floating-point numbers](#floating-point-numbers)

## Bit sequences

The foundation of `base2-mlir` are binary representations of data, described in the following.

### `BitSequence`

An immutable ordered sequence of bits is called a `BitSequence`. It defines a canonical ordering of the bits, which is the MSB->LSB order by convention.

`BitSequence`s are canonically represented as strings of the following format:

```text
bit-sequence ::= `"` bit-literal (trunc-spec)? `"`
bit-literal  ::= `0b` [_01]* | `0x` [_0-9a-fA-F]*
trunc-spec   ::= `/` [0-9]+
```

When **parsing**, the sequence is constructed as follows:

- Accumulate all literal bits in MSB->LSB order.
- If a `trunc-spec` is specified...
  - ...that is greater than the number of bits in the accumulator, remove MSB bits until the length is matched (**truncation**).
  - ...that is smaller than the number of bits in the accumulator, prepend 0 to the MSB until the length is matched (**zero extension**).

When **printing**, the shortest possible encoding is preffered, e.g., `0xF/32` instead of `0x0000000F`.

### `BitSequenceType`

A type that is canonically represented by an ordered sequence of bits is called a `BitSequenceType`.
Instances of this type must be represented using a fixed number of bits, called the `BitWidth`.

The `BitSequenceType` interface declares the following methods:

| Method | Description |
| --: | --- |
| `getBitWidth` | Gets the number of bits required to represent values of this type. |

The `Base2Dialect` automatically registers models for the following built-in types:

- `BFloat16Type`
- `Float16Type`
- `Float32Type`
- `Float64Type`
- `Float80Type`
- `Float128Type`
- `IntegerType`

### `BitSequenceLikeType`

A type that is either a `BitSequenceType` or a `ShapedType` with an element type that is a `BitSequenceType` is a `BitSequenceLikeType`.

### `BitSequenceAttr`

An atribute that declares a compile-time constant value of a `BitSequenceType` using a `BitSequence` is called a `BitSequenceAttr`.

The `BitSequenceAttr` interface declares the following methods:

| Method | Description |
| --: | --- |
| `getBitType()` | Gets the `BitSequenceType`. |
| `getBits()` | Gets the `BitSequence`. |

The `Base2Dialect` automatically registers models for the following built-in attributes:

- `FloatAttr`
- `IntegerAttr`

#### Canonicalization

If possible, a `BitSequenceAttr` will be materialized as a built-in attribute for reasons of readability.
Canonically used are:

- `IntegerAttr` for `IntegerType` with `BitWidth <= 96`
- `FloatAttr` for `FloatType`
- `BitsAttr` for everything else

### `BitSequenceLikeAttr`

An attribute that is either a `BitSequenceAttr` or an `ElementsAttr` with an element type that is a `BitSequenceType` is a `BitSequenceLikeAttr`.

#### Canonicalization

If possible, a `BitSequenceLikeAttr` will be materialized as a built-in `DenseIntOrFPElementsAttr` for reasons of readability.
Otherwise, the `DenseBitsAttr` is used, which is bit-compatible with a `DenseIntElementsAttr` for the same bit width.

### `base2.constant`

A compile-time constant value of a `BitSequenceLikeType` can be materialized from a `BitSequenceLikeAttr` using a `base2.constant` operation.

The order of bits in the bit sequence is semantically tied to the interpretation of the value, not the representation on the target platform.
Lowering to a constant value for a specific target may entail, e.g., conversion of endianness.

> NOTE: Due to MLIR's design of `DenseIntOrFPElementsAttr`, the data layout is actually tied to the host platform layout. For reasons of compatibility and least surprise, `DenseBitsAttr` replicates this behavior. During lowering, the `BitSequence` type and associated free functions provide more precise serialization and deserialization semantics if required.

### `base2.bit_cast`

A `base2.bit_cast` is an operation that reinterpets a value of a `BitSequenceLikeType` as another `BitSequenceLikeType`, without changing the underlying bits.

#### Folding

The `bit_cast` operation is transparent, which means that:

```mlir
%0 = base2.bit_cast %in : T to U
%1 = base2.bit_cast %0 : U to V
```

is equivalent to

```mlir
%1 = base2.bit_cast %in : T to V
```

A `bit_cast` on a constant can always be folded:

```mlir
%0 = base2.constant 0xF : i32
%1 = base2.bit_cast %0 : i32 to si32
```

is equivalent to

```mlir
%1 = base2.constant 0xF : si32
```

## Binary numerals

Numbers in binary representation are special cases of the bit sequences described above.

### `InterpretableType`

A binary numeral is a `BitSequenceType` in which every bit has a fixed semantic, i.e., there is a concrete and unambiguous interpretation for every `BitSequence`.
These types should implement `InterpretableType` to indicate this.

Additionally, compile-time constant `BitSequence`s should be interpretable at compile-time. For that purpose, the `InterpretableType` interface declares the following methods:

| Method | Description |
| --: | --- |
| `canValueCast(inTy,outTy)` | Determine cast legality. |
| `valueCast(inTy,in,outTy,rm)` | Perform constant value cast. |
| `cmp(l,r)` | Perform constant three-way comparison. |
| `min(l,r)` | Compute constant minimum. |
| `max(l,r)` | Compute constant maximum. |
| `add(l,r,rm)` | Compute constant addition. |
| `sub(l,r,rm)` | Compute constant subtraction. |
| `mul(l,r,rm)` | Compute constant multiplication. |
| `div(l,r,rm)` | Perform constant division. |
| `mod(l,r)` | Compute constant modulo. |
| `getFacts(in)` | Infer constant value facts. |

All the above methods have `std::optional`/`bool` return types, where `std::nullopt`/`false` indicates that the operation can not be performed on the given `BitSequence`(s).
Delegating default implementations are provided for `canValueCast`, `min` and `max`.
All other methods have default implementations that return `std::nullopt`/`false`.

This interface is used by the [`BitInterpreter`](BitInterpreter), which is also the recommended way of performing constant folding.

### `InterpretableLikeType`

A type that is either an `InterpretableType` or a `ShapedType` with an element type that is an `InterpretableType` is an `InterpretableLikeType`.

### `base2.value_cast`

A `base2.value_cast` from a value of an `InterpretableLikeType` to a different `InterpretableLikeType` obtains a value of the target type that represents the same value as the input under a given [`RoundingMode`](Rounding) guarantee.

#### Folding

A `value_cast` is generally not foldable, however, the no-op cast can always be elided:

```mlir
%0 = base2.constant -3 : si32
%1 = base2.value_cast %0 : si32 to si32
```

is equivalent to

```mlir
%1 = base2.constant -3 : si32
```

### Common operations

An `InterpretableLikeType` may participate in the following common operations:

| Name | Description |
| --: | --- |
| `cmp` | Compare two numbers. |
| `min` | Obtain the smaller of two numbers. |
| `max` | Obtain the larger of two numbers. |

### Closed arithmetic

An `InterpretableLikeType` may participate in the following closed arithmetic operations:

| Name | Description |
| --: | --- |
| `add` | Computes a sum. |
| `sub` | Computes a difference. |
| `mul` | Computes a product. |
| `div` | Computes a quotient. |
| `mod` | Computes a remainder. |

## Fixed-point numbers

Numbers in which each bit has a constant digit weight are fixed-point numbers.
In `base2`, they must also:

- use two's complement representation for signed numbers
- contain at least the bit with weight $0$ or $1/2$

The `base2` dialect supports this using the `FixedPointSemantics` interface and the [`FixedPointType`](FixedPoint).

## Floating-point numbers

Numbers which are represented using an exponent and a fixed-point mantissa are floating-point numbers.
In `base2`, currently only numbers conforming to the standard set out in IEEE-754 are considered.

The `base2` dialect supports this using the `IEEE754Semantics` interface and the [`IEEE754Type`](IEEE754Type).
