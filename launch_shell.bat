docker build -t gee_env .
docker run --rm -it -v "%CD%:/workdir" gee_env
