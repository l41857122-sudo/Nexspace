@echo off
setlocal EnableDelayedExpansion

REM ==============================================================================
REM   NEXSPACE - AUTOMATED FRESH DEVELOPER ENVIRONMENT SETUP (WINDOWS)
REM   Idempotent setup for Python 3.11 backend, Node.js frontend & local configs
REM ==============================================================================

echo.
echo ==============================================================================
echo   NEXSPACE - DEVELOPER ENVIRONMENT INITIALIZATION
echo ==============================================================================
echo.

cd /d "%~dp0"

REM ------------------------------------------------------------------------------
REM STEP 1: Verify Node.js prerequisite
REM ------------------------------------------------------------------------------
echo [Step 1/6] Checking Node.js runtime...
where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Node.js was not found in your system PATH.
    echo Please install Node.js 18+ or 20+ from https://nodejs.org/
    echo Ensure the option to "Add to PATH" is checked during installation.
    echo.
    exit /b 1
)

for /f "tokens=*" %%i in ('node -v') do set NODE_VERSION=%%i
echo   -- Node.js detected: %NODE_VERSION%

where npm >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] npm was not found in your system PATH.
    echo Please ensure npm is installed alongside Node.js.
    echo.
    exit /b 1
)
for /f "tokens=*" %%i in ('npm -v') do set NPM_VERSION=%%i
echo   -- npm detected:     v%NPM_VERSION%

REM ------------------------------------------------------------------------------
REM STEP 2: Verify Python prerequisite
REM ------------------------------------------------------------------------------
echo.
echo [Step 2/6] Checking Python runtime...

set PYTHON_CMD=
set PY_FOUND=0

REM Check for python in PATH
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do (
        set PY_VER_STR=%%i
        echo   -- Python found in PATH: !PY_VER_STR!
        set PYTHON_CMD=python
        set PY_FOUND=1
    )
)

if !PY_FOUND! equ 0 (
    where py >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        for /f "tokens=*" %%i in ('py -3.11 --version 2^>^&1') do (
            set PY_VER_STR=%%i
            echo   -- Python 3.11 found via py launcher: !PY_VER_STR!
            set PYTHON_CMD=py -3.11
            set PY_FOUND=1
        )
    )
)

if !PY_FOUND! equ 0 (
    where py >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        for /f "tokens=*" %%i in ('py -3 --version 2^>^&1') do (
            set PY_VER_STR=%%i
            echo   -- Python 3 found via py launcher: !PY_VER_STR!
            set PYTHON_CMD=py -3
            set PY_FOUND=1
        )
    )
)

if !PY_FOUND! equ 0 (
    echo.
    echo [ERROR] Python 3.11.x was not found in your PATH or via the py launcher.
    echo Please install Python 3.11.x from https://www.python.org/downloads/
    echo IMPORTANT: Make sure to check "Add Python to PATH" during installation.
    echo.
    exit /b 1
)

REM ------------------------------------------------------------------------------
REM STEP 3: Setup Python Virtual Environment (.venv) [Idempotent]
REM ------------------------------------------------------------------------------
echo.
echo [Step 3/6] Configuring Python Virtual Environment (.venv)...

if exist .venv\Scripts\python.exe goto :VENV_ALREADY_EXISTS

echo   -- Creating fresh virtual environment at .venv ...
%PYTHON_CMD% -m venv .venv
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Failed to create virtual environment with command: %PYTHON_CMD% -m venv .venv
    echo Please ensure venv module is available and permissions are sufficient.
    echo.
    exit /b 1
)
echo   -- Virtual environment successfully created at .venv
goto :VENV_CONFIGURED

:VENV_ALREADY_EXISTS
echo   -- Existing virtual environment found at .venv
echo   -- Preserving existing environment (skipping re-creation).

:VENV_CONFIGURED

set VENV_PYTHON=.venv\Scripts\python.exe
set VENV_PIP=.venv\Scripts\pip.exe

if not exist %VENV_PYTHON% (
    echo.
    echo [ERROR] Virtual environment Python binary not found at %VENV_PYTHON%.
    echo.
    exit /b 1
)

REM ------------------------------------------------------------------------------
REM STEP 4: Install Python Dependencies from requirements.txt
REM ------------------------------------------------------------------------------
echo.
echo [Step 4/6] Installing Python backend dependencies...

echo   -- Upgrading pip in .venv...
"%VENV_PYTHON%" -m pip install --upgrade pip --quiet

if not exist requirements.txt (
    echo.
    echo [ERROR] requirements.txt not found in project root.
    echo.
    exit /b 1
)

echo   -- Installing packages from requirements.txt...
"%VENV_PIP%" install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Failed to install Python dependencies from requirements.txt.
    echo Please check your internet connection and permissions.
    echo.
    exit /b 1
)
echo   -- Python dependencies installed successfully.

REM ------------------------------------------------------------------------------
REM STEP 5: Install Node.js Frontend Dependencies [Idempotent]
REM ------------------------------------------------------------------------------
echo.
echo [Step 5/6] Installing Node.js frontend dependencies (npm install)...

call npm install
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] npm install encountered an error.
    echo Please check the error output above and retry.
    echo.
    exit /b 1
)
echo   -- Node.js dependencies installed successfully.

REM ------------------------------------------------------------------------------
REM STEP 6: Configure Environment (.env.local) [Never Overwrite]
REM ------------------------------------------------------------------------------
echo.
echo [Step 6/6] Verifying environment configuration files...

if exist .env.local goto :ENV_ALREADY_EXISTS

if exist .env.example (
    echo   -- .env.local not found. Initializing from .env.example...
    copy /y .env.example .env.local >nul
    if %ERRORLEVEL% equ 0 (
        echo   -- Created .env.local from .env.example with default ports (3000 / 8000).
    ) else (
        echo   -- [WARNING] Could not copy .env.example to .env.local.
    )
) else (
    echo   -- [WARNING] .env.example was not found.
)
goto :ENV_CONFIGURED

:ENV_ALREADY_EXISTS
echo   -- .env.local already exists.
echo   -- Preserving your active configuration (NO overwrite performed).

:ENV_CONFIGURED

REM ------------------------------------------------------------------------------
REM VERIFICATION: Check critical project files and test imports
REM ------------------------------------------------------------------------------
echo.
echo ==============================================================================
echo   VERIFYING ENVIRONMENT INTEGRITY AND BACKEND CAPABILITIES
echo ==============================================================================

set MISSING_FILES=0
if not exist package.json ( echo   [MISSING] package.json & set MISSING_FILES=1 )
if not exist ml_backend\server.py ( echo   [MISSING] ml_backend\server.py & set MISSING_FILES=1 )
if not exist ml_backend\router.py ( echo   [MISSING] ml_backend\router.py & set MISSING_FILES=1 )
if not exist ml_backend\tools.py ( echo   [MISSING] ml_backend\tools.py & set MISSING_FILES=1 )
if not exist app\page.tsx ( echo   [MISSING] app\page.tsx & set MISSING_FILES=1 )
if not exist .env.local ( echo   [MISSING] .env.local & set MISSING_FILES=1 )

if %MISSING_FILES% neq 0 (
    echo.
    echo [ERROR] One or more critical project files are missing.
    exit /b 1
)

echo   [OK] All critical source and configuration files verified.

echo   -- Testing Python backend module imports in .venv...
"%VENV_PYTHON%" -c "import torch, transformers, PIL, numpy, scipy, tifffile, fastapi, uvicorn; print('  [OK] Python ML modules (torch, transformers, PIL, numpy, scipy, tifffile, fastapi) imported successfully.')"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Python module verification failed in .venv.
    echo.
    exit /b 1
)

echo.
echo ==============================================================================
echo   NEXSPACE SETUP COMPLETED SUCCESSFULLY!
echo ==============================================================================
echo.
echo Next Steps:
echo.
echo   1. Activate the Python virtual environment in your terminal:
echo        .venv\Scripts\activate
echo.
echo   2. Launch the full live application (Next.js + FastAPI):
echo        npm run demo
echo      or concurrently:
echo        npm run dev
echo.
echo   3. Run automated test suites:
echo        python ml_backend/test_step10_hardening.py
echo        node test_e2e_live_integration.js
echo.
echo ==============================================================================
echo.

exit /b 0
