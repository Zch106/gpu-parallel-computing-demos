@echo off
REM ============================================================
REM  Windows 一键编译（无需 make / 无需手动开 VS 命令行）
REM  要求：已安装 CUDA Toolkit（nvcc 在 PATH）和 Visual Studio（含 C++ 桌面开发负载）
REM  脚本会自动用 vswhere 定位 vcvarsall.bat 并初始化 x64 编译环境
REM ============================================================
setlocal

REM --- 定位 vcvarsall.bat ---
set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSINST="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -property installationPath`) do set "VSINST=%%i"
)
if not defined VSINST (
    echo [ERROR] 未通过 vswhere 找到 Visual Studio。请确认已安装 "使用 C++ 的桌面开发" 工作负载。
    exit /b 1
)
echo [INFO] 使用 Visual Studio 安装路径: %VSINST%
call "%VSINST%\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
    echo [ERROR] vcvarsall.bat 初始化失败。
    exit /b 1
)

REM --- 编译参数 ---
set NVCC_FLAGS=-arch=sm_86 -Iinclude -Xcompiler /utf-8
set SRCS=src\main.cu src\cpu_ref.cu src\softmax.cu src\gemm_naive.cu src\gemm_tiled.cu src\gemm_opt.cu src\gemm_tc.cu src\cublas_ref.cu src\report.cu

echo [INFO] 开始编译 (nvcc %NVCC_FLAGS%) ...
nvcc %NVCC_FLAGS% %SRCS% -o attention_demo.exe -lcublas
if %errorlevel%==0 (
    echo.
    echo Build OK. 运行：attention_demo.exe
) else (
    echo.
    echo Build FAILED.
    exit /b 1
)
endlocal
