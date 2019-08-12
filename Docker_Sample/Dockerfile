FROM continuumio/anaconda3
MAINTAINER ACE, https://markacebedo.netlify.com
COPY ./flask_demo /usr/local/python/
EXPOSE 5000
WORKDIR /usr/local/python/
RUN pip install -r requirements.txt
CMD python flask_api.py
