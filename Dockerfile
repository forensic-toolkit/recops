FROM python:3.10

WORKDIR /src

ARG     SKIP_MODELS_DOWNLOAD=true

RUN     mkdir -p /src/bin
COPY    bin/recops-download-models.sh   /src/bin/recops-download-models.sh
RUN     bash -c 'if [ $SKIP_MODELS_DOWNLOAD ]; then echo skip model download; else bash /src/bin/recops-download-models.sh;fi'

RUN 	apt-get update &&\
		apt-get install -y --no-install-recommends \
			libgl1 \
			libglib2.0-0

COPY    requirements.txt /src
RUN     pip install -r requirements.txt 

COPY    bin/recops /src/bin/recops
COPY    recops     /src/recops
COPY    setup.py   /src/setup.py
COPY    LICENSE.md /src/LICENSE.md
COPY    README.md  /src/README.md

RUN     python setup.py install

EXPOSE  5000

RUN     mkdir -p /var/lib/recops

VOLUME  /var/lib/recops

ENV     STORAGE_URI=file:///var/lib/recops/data 
ENV     DATABASE_URI=sqlite:////var/lib/recops/recops.db 

ENTRYPOINT ["/usr/local/bin/recops"]
CMD        ["--help"]