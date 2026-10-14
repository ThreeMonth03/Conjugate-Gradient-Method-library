cmake_minimum_required(VERSION 3.24)

# Prefix every symbol defined by the static provider, including its internal
# references. libc/pthread references remain untouched. Upstream SYMBOLPREFIX
# handles shared exports, so a static archive needs this explicit namespace.
set(archive "${CG_BLAS_BINARY}/lib/libopenblas.a")
if(NOT EXISTS "${archive}")
    message(FATAL_ERROR "Private OpenBLAS archive missing: ${archive}")
endif()
execute_process(COMMAND "${CG_BLAS_NM}" -g --defined-only "${archive}"
    OUTPUT_VARIABLE symbols COMMAND_ERROR_IS_FATAL ANY)
string(REGEX MATCHALL "[0-9a-fA-F]+ [A-Za-z] [^ \n\r]+" definitions "${symbols}")
set(names)
foreach(definition IN LISTS definitions)
    string(REGEX REPLACE "^[0-9a-fA-F]+ [A-Za-z] " "" name "${definition}")
    list(APPEND names "${name}")
endforeach()
list(REMOVE_DUPLICATES names)
list(SORT names)
if(NOT "cblas_dgemv" IN_LIST names)
    message(FATAL_ERROR "Private OpenBLAS archive does not define cblas_dgemv")
endif()
set(mapping "")
foreach(name IN LISTS names)
    string(APPEND mapping "${name} cgpy_${name}\n")
endforeach()
file(MAKE_DIRECTORY "${CG_BLAS_PRIVATE}/lib" "${CG_BLAS_PRIVATE}/include")
file(WRITE "${CG_BLAS_PRIVATE}/symbols.map" "${mapping}")
execute_process(COMMAND "${CG_BLAS_OBJCOPY}"
    --redefine-syms "${CG_BLAS_PRIVATE}/symbols.map"
    "${archive}" "${CG_BLAS_PRIVATE}/lib/libcgpy_openblas.a"
    COMMAND_ERROR_IS_FATAL ANY)
file(COPY "${CG_BLAS_BINARY}/generated/cblas.h" "${CG_BLAS_BINARY}/openblas_config.h"
    DESTINATION "${CG_BLAS_PRIVATE}/include")
file(READ "${CG_BLAS_PRIVATE}/include/openblas_config.h" config)
if(config MATCHES "#define OPENBLAS_USE64BITINT")
    message(FATAL_ERROR "Private OpenBLAS must use 32-bit BLAS integers")
endif()
