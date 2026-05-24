@echo off
setlocal

:: This build script compiles the CUDA file into a shared library (DLL)
:: instead of an executable, so it can be loaded into Python via ctypes.

set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"
set "CCBIN=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64"
set "SRC=trajectory_math.cu"
set "OUT=..\trajectory.dll"

echo =======================================================
echo Building %SRC% into %OUT% ...
echo =======================================================

"%NVCC%" --shared --use_fast_math -lineinfo -std=c++17 -O2 -ccbin "%CCBIN%" -Xcompiler "/EHsc /openmp /MD" "%SRC%" -o "%OUT%"

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] Successfully built %OUT%!
    echo You can now use it in your PySpark notebook.
) else (
    echo.
    echo [ERROR] Build failed!
)
