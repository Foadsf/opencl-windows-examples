@echo off
setlocal

set CMAKE="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

if not exist build mkdir build
cd build
%CMAKE% .. -G "Visual Studio 16 2019" -A x64
if errorlevel 1 (
    cd ..
    pause
    exit /b 1
)

%CMAKE% --build . --config Release
if errorlevel 1 (
    cd ..
    pause
    exit /b 1
)

copy ..\nbody.cl Release\nbody.cl >nul

cd ..
echo.
echo Running N-body simulation comparison...
cd build\Release
nbody_simulation.exe
cd ..\..
pause