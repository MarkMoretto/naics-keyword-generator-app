@ECHO Off

CD client-side

SET cmd=yarn start

START "client-side" %ComSpec% /c %cmd%

GOTO :eof
