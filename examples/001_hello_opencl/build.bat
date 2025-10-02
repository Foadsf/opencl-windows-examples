@echo off
setlocal

set CMAKE="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

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

REM Copy kernel file to Release directory
copy hello.cl Release\hello.cl >nul

cd ..
echo.
echo Running hello_opencl...
if exist build\Release\hello_opencl.exe (
    cd build\Release
    hello_opencl.exe
    cd ..\..
) else (
    echo Executable not found!
)
pause