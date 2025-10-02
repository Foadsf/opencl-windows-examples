@echo off
setlocal enabledelayedexpansion

echo ============================================
echo OpenCL Hardware Detection
echo ============================================
echo.

echo [CPU Information]
wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors /format:table
echo.

echo [GPU Information]
wmic path win32_VideoController get Name,AdapterRAM,DriverVersion /format:table
echo.

echo [System Memory]
wmic ComputerSystem get TotalPhysicalMemory /format:list
echo.

echo ============================================
echo Checking for OpenCL Support...
echo ============================================
echo.

REM Check if clinfo is available
where clinfo >nul 2>&1
if %errorlevel% equ 0 (
    echo Running clinfo to detect OpenCL platforms and devices:
    echo.
    clinfo
) else (
    echo clinfo not found. Install it with: choco install opencl-sdk
    echo or we'll install the OpenCL SDK next.
)

echo.
echo ============================================
echo Detection Complete
echo ============================================
pause