add_library(BitInterfaces OBJECT
    BitSequenceAttr.cpp
    BitSequenceType.cpp
    LogicOp.cpp
)
add_dependencies(BitAnalysis BitIncGen)
