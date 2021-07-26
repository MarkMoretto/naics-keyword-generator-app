@ECHO Off



::START "flask" %comspec% /c start-flask.cmd
CALL start-server.cmd 
ping 127.0.0.1 -l 1 -n 3
CALL launch-browser.cmd
GOTO :eof

