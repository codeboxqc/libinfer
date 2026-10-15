
E:\vcpkg>vcpkg install libinfer --overlay-ports="vcpkg-custom-ports" --head
Computing installation plan...
The following packages will be built and installed:
    libinfer:x64-windows@0.1.0 (+HEAD) -- E:\vcpkg\vcpkg-custom-ports\libinfer
Installing 1/1 libinfer:x64-windows@0.1.0...
Building libinfer:x64-windows@0.1.0 from HEAD...
E:\vcpkg\vcpkg-custom-ports\libinfer: info: installing overlay port from here
Downloading https://api.github.com/repos/codeboxqc/libinfer/git/refs/heads/main -> codeboxqc-libinfer-main.version
Successfully downloaded codeboxqc-libinfer-main.version
Downloading https://github.com/codeboxqc/libinfer/archive/d7b64edc2e91fe13de4047ca47f97b4253f9dc28.tar.gz -> codeboxqc-libinfer-d7b64edc2e91fe13de4047ca47f97b4253f9dc28.tar.gz
Successfully downloaded codeboxqc-libinfer-d7b64edc2e91fe13de4047ca47f97b4253f9dc28.tar.gz
-- Extracting source E:/vcpkg/downloads/codeboxqc-libinfer-d7b64edc2e91fe13de4047ca47f97b4253f9dc28.tar.gz
-- Using source at E:/vcpkg/buildtrees/libinfer/src/head/4253f9dc28-054f169820.clean
-- Configuring x64-windows
-- Building x64-windows-dbg
CMake Warning at scripts/cmake/vcpkg_execute_build_process.cmake:65 (message):
  Please ensure your system has sufficient memory.
Call Stack (most recent call first):
  installed/x64-windows/share/vcpkg-cmake/vcpkg_cmake_build.cmake:74 (vcpkg_execute_build_process)
  installed/x64-windows/share/vcpkg-cmake/vcpkg_cmake_install.cmake:16 (vcpkg_cmake_build)
  vcpkg-custom-ports/libinfer/portfile.cmake:13 (vcpkg_cmake_install)
  scripts/ports.cmake:209 (include)


-- Restarting build without parallelism
CMake Error at scripts/cmake/vcpkg_execute_build_process.cmake:134 (message):
    Command failed: E:/vcpkg/downloads/tools/cmake-4.4.3-windows/cmake-4.4.3-windows-x86_64/bin/cmake.exe --build . --config Debug --target install -- -v -j13
    Working Directory: E:/vcpkg/buildtrees/libinfer/x64-windows-dbg
    See logs for more information:
      E:\vcpkg\buildtrees\libinfer\install-x64-windows-dbg-out.log
      E:\vcpkg\buildtrees\libinfer\install-x64-windows-dbg-out-1.log

Call Stack (most recent call first):
  installed/x64-windows/share/vcpkg-cmake/vcpkg_cmake_build.cmake:74 (vcpkg_execute_build_process)
  installed/x64-windows/share/vcpkg-cmake/vcpkg_cmake_install.cmake:16 (vcpkg_cmake_build)
  vcpkg-custom-ports/libinfer/portfile.cmake:13 (vcpkg_cmake_install)
  scripts/ports.cmake:209 (include)


error: building libinfer:x64-windows failed with: BUILD_FAILED
See https://learn.microsoft.com/vcpkg/troubleshoot/build-failures?WT.mc_id=vcpkg_inproduct_cli for more information.
Elapsed time to handle libinfer:x64-windows: 6.3 s
Please ensure you're using the latest port files with `git pull` and `vcpkg update`.
Then check for known issues at:
  https://github.com/microsoft/vcpkg/issues?q=is%3Aissue+is%3Aopen+in%3Atitle+libinfer
You can submit a new issue at:
  https://github.com/microsoft/vcpkg/issues/new?title=%5Blibinfer%5D%20build%20error%20on%20x64-windows&body=Copy%20issue%20body%20from%20E%3A%2Fvcpkg%2Finstalled%2Fvcpkg%2Fissue_body.md
