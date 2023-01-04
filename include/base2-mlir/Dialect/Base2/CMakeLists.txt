################################################################################
# Base2IncGen
#
# MLIR Base2 dialect generated includes.
################################################################################

add_custom_target(Base2IncGen)

mlir_gen_enums(Base2)

# Interfaces.
add_subdirectory(Interfaces)
# Attributes, Dialect, Operations and Types.
add_subdirectory(IR)
