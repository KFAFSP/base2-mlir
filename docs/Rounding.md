# Rounding

- [Rounding](#rounding)
  - [Rounding modes](#rounding-modes)
    - [Integer](#integer)
    - [Fixed-point](#fixed-point)
    - [Floating-point](#floating-point)

Rounding occurs when an exact (mathematical) result can not be represented by the result type, causing it to assume an approximate value.
Operations that may incur rounding are:

- `base2.value_cast`
- `base2.add`, `base2.sub`, `base2.mul`, `base2.div`

These operations require a `RoundingMode` attribute to be defined, which indicates how rounding shall be performed.
If unspecified, `RoundingMode::None` is assumed, which names an implementation defined mode that shall be the fastest.

## Rounding modes

The `RoundingMode` enumeration lists all known rounding modes.
Given an input value `i` and a rounding result `o`, the rounding modes are defined by guarantees on `o`, which are:

|           Mode | Guarantee |
| -------------: | --- |
|         `None` | None *(fastest)*. |
|      `Nearest` | $\min(\lvert i - o\rvert)$ |
|      `RoundUp` | $\min(\lvert i - o\rvert)$ with $o \ge i$ |
|    `RoundDown` | $\min(\lvert i - o\rvert)$ with $o \le i$ |
|  `TowardsZero` | $\min(\lvert i - o\rvert)$ with $\lvert o\rvert \le \lvert i\rvert$ |
| `AwayFromZero` | $\min(\lvert i - o\rvert)$ with $\lvert o\rvert \ge \lvert i\rvert$ |
|     `Converge` | $\min(\lvert i - o\rvert)$ with $\lim_{x\rightarrow\infty}(i_x-o_x) = 0$ |

### Integer

On integer operations, the rounding modes instead refer to the saturation behavior on arithmetic under- and overflow.
The signedness of the operands only impacts the `min` and `max` limits in this table.

For an integer type of width $N$, these behaviors are:

|           Mode |  underflow  |   overflow  |
| -------------: | :---------: | :---------: |
|         `None` | $\mod  2^N$ | $\mod  2^N$ |
|      `Nearest` |    `min`    |    `max`    |
|      `RoundUp` |    `min`    |    *I.d.*   |
|    `RoundDown` |    *I.d.*   |    `max`    |
|  `TowardsZero` |    `min`    | $\mod  2^N$ |
| `AwayFromZero` | $\mod  2^N$ |    `max`    |
|     `Converge` |    *I.d.*   |    *I.d.*   |

### Fixed-point

For overflow behavior, the same rules apply as for integer rounding.

For a fixed-point type of width $I.F$, the rounding behaviors are:

|           Mode | Behavior |
| -------------: | --- |
|         `None` | *Implementation defined* |
|      `Nearest` | $(i + 2^{F-1} - \texttt{signBit}) >> F$ |
|      `RoundUp` | $(i + 2^F-1) >> F$ |
|    `RoundDown` | $i >> F$ |
|  `TowardsZero` | $\begin{cases}i >> F&i\ge 0\\(i + 2^F-1)>>F&i<0\end{cases}$ |
| `AwayFromZero` | $\begin{cases}i >> F&i<0\\(i + 2^F-1)>>F&i\ge0\end{cases}$ |
|     `Converge` | *Implementation defined* |

### Floating-point

The above rounding modes are mapped onto the **IEEE-754** modes as follows:

|           Mode | **IEEE-754** |
| -------------: | --- |
|         `None` | *Implementation defined* |
|      `Nearest` | Round to nearest, ties to even |
|      `RoundUp` | Round to $+\infty$ |
|    `RoundDown` | Round to $-\infty$ |
|  `TowardsZero` | Round toward 0 |
| `AwayFromZero` | Round to nearest, ties away from zero |
|     `Converge` | *Implementation defined* |
