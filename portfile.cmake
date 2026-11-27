vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO codeboxqc/libinfer
    REF v0.1.0 
    SHA512 0 
    HEAD_REF main
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

# Move headers to the correct vcpkg directory
vcpkg_cmake_config_fixup(PACKAGE_NAME libinfer)

# Install your license file
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)