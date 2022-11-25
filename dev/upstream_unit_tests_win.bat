@ECHO ON
pushd %TEMP% || goto :error
cd \conda_src || goto :error
CALL dev-init.bat || goto :error
CALL conda info || goto :error
CALL python -c "from importlib_metadata import version; print('libmambapy', version('libmambapy'))" || goto :error
CALL python -m pytest -v tests/core/test_solve.py || goto :error
goto :EOF

:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%
