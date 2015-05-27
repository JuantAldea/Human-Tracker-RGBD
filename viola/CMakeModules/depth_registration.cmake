 include(ExternalProject)
 ExternalProject_Add (depth_registration_lib
    URL "${viola_SOURCE_DIR}/depth_registration"
    UPDATE_COMMAND ""
   INSTALL_COMMAND ""
)

ExternalProject_Get_Property(depth_registration_lib source_dir)
set(DEPTH_REGISTRATION_INCLUDE_DIR ${source_dir}/include)

ExternalProject_Get_Property(depth_registration_lib binary_dir)
set(DEPTH_REGISTRATION_LIBRARY_PATH ${binary_dir}/${CMAKE_FIND_LIBRARY_PREFIXES}depth_registration.so)

set(DEPTH_REGISTRATION_LIBRARY depth_registration)

add_library(${DEPTH_REGISTRATION_LIBRARY} UNKNOWN IMPORTED)

set_property(TARGET ${DEPTH_REGISTRATION_LIBRARY} PROPERTY IMPORTED_LOCATION ${DEPTH_REGISTRATION_LIBRARY_PATH})

add_dependencies(${DEPTH_REGISTRATION_LIBRARY} depth_registration_lib)