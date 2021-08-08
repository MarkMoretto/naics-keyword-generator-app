@ECHO Off

:: https://create-react-app.dev/docs/advanced-configuration
SET BROWSER=none


CD client-side

SET cmd=yarn start
SET CHOKIDAR_USEPOLLING=true
SET FAST_REFRESH=false
::SET GENERATE_SOURCEMAP=true


START "client-side" %ComSpec% /c %cmd%


GOTO :eof
