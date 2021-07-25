@ECHO Off


SET PYRO_SERIALIZERS_ACCEPTED=pickle
SET PYRO_SERIALIZER=pickle

CD server-side

SET cmd=python -m Pyro4.naming

CALL venv\Scripts\activate && %cmd%
GOTO :eof
