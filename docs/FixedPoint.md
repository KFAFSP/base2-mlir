# FixedPoint

- [FixedPoint](#fixedpoint)
  - [`FixedPointSemantics`](#fixedpointsemantics)
    - [Superset relation](#superset-relation)
  - [Operations](#operations)

An integer with an assumed scaling (virtual decimal point) is called a fixed-point number, and can be used to represent rational numbers.

## `FixedPointSemantics`

An instance of a fixed-point type is described by its `FixedPointSemantics`.
These are defined by:

|        Parameter | Description |
| ---------------: | --- |
|     `Signedness` | Signedness semantics. |
|    `IntegerBits` | Number of integer bits. |
| `FractionalBits` | Number of fractional bits. |

For `FractionalBits` $=0$, the fixed-point type is semantically equivalent to its underlying `IntegerType`.

### Superset relation

Given two semantics `this` and `other`, it can be determined whether every value of `other` can be encoded in `this` without rounding.
This property defines the superset relation `this` $\ge$ `other`.

- If `this` and `other` have the same signedness, then:

  ```text
  this >= other <=> this.IntBits >= other.IntBits
                    and this.FracBits >= other.FracBits
  ```

- If `this` is signed and `other` is unsigned, then:

  ```text
  this >= other <-- this.IntBits > other.IntBits
                    and this.FracBits >= other.FracBits
  this >= other <-- this.IntBits == 0 == other.IntBits
                    and this.FracBits > other.FracBits
  ```

- If `this` is unsigned and `other` is signed, then `this` $<$ `other`.

## Operations

> TODO: Port these over again.
