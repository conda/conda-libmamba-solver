@ECHO ON
pushd %TEMP% || goto :error
cd \conda_src || goto :error
CALL dev-init.bat || goto :error
CALL conda info || goto :error
CALL mamba --version || goto :error
CALL python -m pytest -v tests/core/test_solve.py || goto :error
goto :EOF

:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%