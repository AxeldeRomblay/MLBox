FROM localhost:8992/python:2.7

COPY *.whl /tmp/

RUN pip install /tmp/*.whl

RUN rm /tmp/*.whl

CMD ["python2"]
