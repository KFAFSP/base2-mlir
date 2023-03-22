add_library(BitInterfaces OBJECT
    BitSequenceAttr.cpp
    BitSequenceType.cpp
)
add_dependencies(BitAnalysis BitIncGen)
