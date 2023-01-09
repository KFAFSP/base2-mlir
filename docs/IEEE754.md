# IEEE-754

- [IEEE-754](#ieee-754)
  - [`IEEE754Semantics`](#ieee754semantics)
    - [IEEE-754 parametrization](#ieee-754-parametrization)
    - [Superset relation](#superset-relation)
  - [Operations](#operations)

The [IEEE-754](https://standards.ieee.org/ieee/754/6210/) specification defines a standard for representing floating-point numbers in a binary format, and operations on them.

## `IEEE754Semantics`

The IEEE-754 standard defines a family of types, and names some concrete instances (single, double, ...).
We interpret the standard to limit this family to types which:

- Have a signed-magnitude representation (**sign bit**)
- Have a special maximum biased exponent (**NaN** and **Inf**)
- Adopt the hidden-one notation (**denormalization** exists)

We allow naming arbitrary instances using `IEEE754Semantics`.
These are defined by:

|      Parameter | Description |
| -------------: | --- |
|    `Precision` | Number of mantissa bits $+1$. |
| `ExponentBits` | Number of exponent bits. |
|         `Bias` | Exponent bias value. |

Such that the given combination of parameters describes a valid type (bit width limit, bias fits).

The representable minimum exponent value is then $1-$ `Bias`, and the maximum exponent value is $(1 << \texttt{expBits}) - 2 - \texttt{bias}$.

### IEEE-754 parametrization

The IEEE-754 standard constructs its named types using a simplified scheme, where the parameters are:

|     Parameter | Description |
| ------------: | --- |
|   `Precision` | Number of mantissa bits $+1$. |
| `MaxExponent` | Maximum exponent value. |
| `MinExponent` | Minimum exponent value. |

Additionally, `MinExponent` is chosen to be $1-$ `MaxExponent`.
Under this scheme, the `Bias` is constructed via $1-$ `MinExponent`.
Then, a number of `ExponentBits` is chosen such that `MaxExponent` $+$ `Bias` $+1$ fits.

### Superset relation

Given two semantics `this` and `other`, it can be determined whether every value of `other` can be encoded in `this` without rounding.
This property defines the superset relation `this` $\ge$ `other`:

```text
this >= other <=> this.Precision >= other.Precision
                  and this.MinExponent <= other.MinExponent
                  and this.MaxExponent >= other.MaxExponent
```

## Operations

> TODO: Port these over again.
