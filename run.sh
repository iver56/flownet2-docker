#!/usr/bin/env bash

source /flownet2/flownet2/set-env.sh

uwsgi --http :5003 --wsgi-file web_service.py --callable app --enable-threads
