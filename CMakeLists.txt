
E:\vcpkg>vcpkg install libinfer --overlay-ports="vcpkg-custom-ports" --head
Computing installation plan...
The following packages will be built and installed:
    libinfer:x64-windows@0.1.0 (+HEAD) -- E:\vcpkg\vcpkg-custom-ports\libinfer
Installing 1/1 libinfer:x64-windows@0.1.0...
Building libinfer:x64-windows@0.1.0 from HEAD...
E:\vcpkg\vcpkg-custom-ports\libinfer: info: installing overlay port from here
Downloading https://api.github.com/repos/codeboxqc/libinfer/git/refs/heads/main -> codeboxqc-libinfer-main.version
Successfully downloaded codeboxqc-libinfer-main.version
Downloading https://github.com/codeboxqc/libinfer/archive/07fc7f9351e0311898e3390bbf99ab09f7e06571.tar.gz -> codeboxqc-libinfer-07fc7f9351e0311898e3390bbf99ab09f7e06571.tar.gz
Successfully downloaded codeboxqc-libinfer-07fc7f9351e0311898e3390bbf99ab09f7e06571.tar.gz
-- Extracting source E:/vcpkg/downloads/codeboxqc-libinfer-07fc7f9351e0311898e3390bbf99ab09f7e06571.tar.gz
-- Using source at E:/vcpkg/buildtrees/libinfer/src/head/09f7e06571-1ddb43bd55.clean
-- Configuring x64-windows
CMake Error at scripts/cmake/vcpkg_execute_required_process.cmake:127 (message):
    Command failed: E:\\vcpkg\\downloads\\tools\\ninja-1.13.2-windows\\ninja.exe -v
    Working Directory: E:/vcpkg/buildtrees/libinfer/x64-windows-rel/vcpkg-parallel-configure
    Error code: 1
    See logs for more information:
      E:\vcpkg\buildtrees\libinfer\config-x64-windows-dbg-CMakeCache.txt.log
      E:\vcpkg\buildtrees\libinfer\config-x64-windows-rel-CMakeCache.txt.log
      E:\vcpkg\buildtrees\libinfer\config-x64-windows-out.log

Call Stack (most recent call first):
  installed/x64-windows/share/vcpkg-cmake/vcpkg_cmake_configure.cmake:271 (vcpkg_execute_required_process)
  vcpkg-custom-ports/libinfer/portfile.cmake:9 (vcpkg_cmake_configure)
  scripts/ports.cmake:209 (include)


error: building libinfer:x64-windows failed with: BUILD_FAILED
See https://learn.microsoft.com/vcpkg/troubleshoot/build-failures?WT.mc_id=vcpkg_inproduct_cli for more information.
Elapsed time to handle libinfer:x64-windows: 2.5 s
Please ensure you're using the latest port files with `git pull` and `vcpkg update`.
Then check for known issues at:
  https://github.com/microsoft/vcpkg/issues?q=is%3Aissue+is%3Aopen+in%3Atitle+libinfer
You can submit a new issue at:
  https://github.com/microsoft/vcpkg/issues/new?title=%5Blibinfer%5D%20build%20error%20on%20x64-windows&body=Copy%20issue%20body%20from%20E%3A%2Fvcpkg%2Finstalled%2Fvcpkg%2Fissue_body.md
