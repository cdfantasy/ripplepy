set(PSPLINE_SOURCES
    pspline/r8akherm3.f
    pspline/r8akherm1.f
    pspline/ibc_ck.f
    pspline/r8splinck.f
    pspline/r8herm3ev.f
    pspline/r8zonfind.f
    pspline/stel_kinds.f
)

set(OTHER_SOURCES
    DLSODE.f
    hybrd.f
)

add_library(ripple_fortran_obj OBJECT
    ripple.f90
    ${OTHER_SOURCES}
    ${PSPLINE_SOURCES}
)

set_source_files_properties(${PSPLINE_SOURCES} ${OTHER_SOURCES} PROPERTIES
    COMPILE_FLAGS "${F77_FLAGS_RELAX}"
)

# gfortran 13 auto-vectorizes atan2 into libmvec calls (_ZGVbN2vv_atan2) at -O3.
# The f2py link step does not link libmvec, so on glibc builds where libmvec is a
# separate library (e.g. Ubuntu 24.04) this yields an undefined symbol at import.
# Disable vectorization for this file only to avoid emitting that symbol.
set_source_files_properties(ripple.f90 PROPERTIES
    COMPILE_FLAGS "-fno-tree-vectorize"
)


target_include_directories(ripple_fortran_obj PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/pspline
)