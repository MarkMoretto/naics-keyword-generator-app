@echo off

CD client-side

START "sass" %ComSpec% /k "(yarn sass:watch)"

GOTO :EOF
