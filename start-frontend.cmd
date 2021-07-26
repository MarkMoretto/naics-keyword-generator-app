@ECHO Off

:: https://create-react-app.dev/docs/advanced-configuration
SET BROWSER=none


CD client-side

SET cmd=yarn start

START "client-side" %ComSpec% /c %cmd%

GOTO :eof
