add_library(Base2Interfaces OBJECT
    ArithmeticOp.cpp
    FixedPointSemantics.cpp
    IEEE754Semantics.cpp
    InterpretableType.cpp
    RoundingOp.cpp
)
add_dependencies(Base2Interfaces Base2IncGen)
