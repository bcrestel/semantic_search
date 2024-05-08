FROM piptools:latest

ADD requirements.txt ./
RUN pip install -r requirements.txt

# Add jupyterlab extensions
RUN pip install jupyterlab_execute_time

ENV PYTHONPATH="/home"