@ECHO Off



::START "flask" %comspec% /c start-flask.cmd
::CALL start-frontend.cmd
::PING -t 127.0.0.1 -l 1 -n 5 1>nul 2>&1
::CALL start-server.cmd
::ping 127.0.0.1 -l 1 -n 3 2>nul
::CALL launch-browser.cmd
::GOTO :eof


START "frontend" %ComSpec% /C start-frontend.cmd
::PING -t 127.0.0.1 -l 1 -n 5 1>nul 2>&1
START "backend" %ComSpec% /C start-server.cmd

GOTO :eof