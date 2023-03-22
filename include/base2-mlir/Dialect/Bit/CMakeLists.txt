################################################################################
# BitIncGen
#
# MLIR Bit dialect generated includes.
################################################################################

add_custom_target(BitIncGen)

mlir_gen_enums(Bit)

# Interfaces.
add_subdirectory(Interfaces)
# Attributes, Dialect, Operations and Types.
add_subdirectory(IR)
