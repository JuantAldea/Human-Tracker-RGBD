GET_FILENAME_COMPONENT(LIBUSB_DIR "${viola_SOURCE_DIR}/depends/libusb/" REALPATH)
INCLUDE_DIRECTORIES("${LIBUSB_DIR}/include/libusb-1.0/")
LINK_DIRECTORIES("${LIBUSB_DIR}/lib/")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)