@echo off
setlocal

REM Use the MSVC bundled CMake
set CMAKE="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

if not exist %CMAKE% (
    echo CMake not found at expected location!
    echo Please install CMake or update the path in this script.
    pause
    exit /b 1
)

if not exist build mkdir build
cd build
%CMAKE% .. -G "Visual Studio 16 2019" -A x64
if errorlevel 1 (
    echo CMake configuration failed!
    cd ..
    pause
    exit /b 1
)

%CMAKE% --build . --config Release
if errorlevel 1 (
    echo Build failed!
    cd ..
    pause
    exit /b 1
)

cd ..
echo.
echo Running device enumeration...
if exist build\Release\device_enumeration.exe (
    build\Release\device_enumeration.exe
) else (
    echo Executable not found!
)
pause