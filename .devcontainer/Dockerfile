FROM docker.io/jupyter/datascience-notebook:latest

USER root
# Generally, Dev Container Features assume that the non-root user (in this case jovyan)
# is in a group with the same name (in this case jovyan). So we must first make that so.
RUN groupadd jovyan \
    && usermod -g jovyan -a -G users jovyan

RUN apt-get update && apt-get install ripgrep

USER jovyan
