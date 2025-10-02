@echo off
setlocal

set CMAKE="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

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

copy vector_add.cl Release\vector_add.cl >nul

cd ..
echo.
echo Running breakeven analysis (this will take a minute)...
if exist build\Release\breakeven_analysis.exe (
    cd build\Release
    breakeven_analysis.exe
    cd ..\..
) else (
    echo Executable not found!
)
pause