@echo off
setlocal enabledelayedexpansion

echo ============================================
echo OpenCL Installation Status Check
echo ============================================
echo.

echo [Checking OpenCL Runtime Libraries]
if exist "C:\Windows\System32\OpenCL.dll" (
    echo [OK] OpenCL.dll found in System32
    dir "C:\Windows\System32\OpenCL.dll" | findstr /C:"OpenCL.dll"
) else (
    echo [MISSING] OpenCL.dll not found in System32
)

if exist "C:\Windows\SysWOW64\OpenCL.dll" (
    echo [OK] OpenCL.dll found in SysWOW64 ^(32-bit^)
    dir "C:\Windows\SysWOW64\OpenCL.dll" | findstr /C:"OpenCL.dll"
) else (
    echo [MISSING] OpenCL.dll not found in SysWOW64
)
echo.

echo [Checking NVIDIA CUDA Installation]
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" (
    echo [OK] NVIDIA CUDA Toolkit found
    dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" /B
) else (
    echo [MISSING] NVIDIA CUDA Toolkit not found
)
echo.

echo [Checking Intel OpenCL Runtime]
if exist "C:\Program Files (x86)\Intel\OpenCL" (
    echo [OK] Intel OpenCL found
    dir "C:\Program Files (x86)\Intel\OpenCL" /B
) else (
    echo [MISSING] Intel OpenCL not found at standard location
)

if exist "C:\Program Files\Intel\OpenCL" (
    echo [OK] Intel OpenCL found ^(64-bit location^)
    dir "C:\Program Files\Intel\OpenCL" /B
) else (
    echo [NOTE] Intel OpenCL not found at alternate location
)
echo.

echo [Checking Chocolatey Installed Packages]
where choco >nul 2>&1
if %errorlevel% equ 0 (
    echo Checking for opencl-intel-cpu-runtime...
    choco list --local-only | findstr /I "opencl-intel"
    echo.
    echo Checking for CUDA...
    choco list --local-only | findstr /I "cuda"
) else (
    echo Chocolatey not found
)
echo.

echo ============================================
echo Recommendations
echo ============================================
pause