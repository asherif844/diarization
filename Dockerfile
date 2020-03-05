# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

FROM python:3.6-slim

LABEL maintainer="anthony.franklin@microsoft.com"

WORKDIR /opt/app
RUN mkdir /opt/app/data


# Install Common Dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends --no-install-suggests ffmpeg && \ 
    apt-get install -y --no-install-recommends libmagic1 && rm -rf /var/lib/apt/lists/*


# Install jupyter notebook
# RUN pip install --no-cache-dir jupyter

# Copy Diarization app into container and setup requirements
COPY . /opt/app
RUN pip install --no-cache-dir -r requirements.txt


# Clean up install caches
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Expose container ports
EXPOSE 5000


# Define external mount points
VOLUME /opt/app/data
VOLUME /opt/app/data/input
ENTRYPOINT ["python"]
CMD ["app.py"]

# CMD [ "gunicorn" , "-b", "0.0.0.0:80", "app:app", "--reload" ]
