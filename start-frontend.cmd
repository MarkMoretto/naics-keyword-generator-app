@ECHO Off

CD client-side

SETLOCAL

:: https://create-react-app.dev/docs/advanced-configuration
SET BROWSER=none
::SET CHOKIDAR_USEPOLLING=true
SET FAST_REFRESH=false
::SET GENERATE_SOURCEMAP=true


SET cmd=yarn start

START "client-side" %ComSpec% /c %cmd%


GOTO :eof
