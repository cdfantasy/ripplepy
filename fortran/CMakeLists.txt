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

target_include_directories(ripple_fortran_obj PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/pspline
)