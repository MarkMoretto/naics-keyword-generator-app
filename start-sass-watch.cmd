@echo off

CD client-side

START "sass" %ComSpec% /c "(yarn sass:watch)"

GOTO :EOF
