# -*- coding: utf-8 -*-
import os
import signal
import subprocess
import sys

import multiprocessing
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from recsys.ap_recsys import ApRecsys
from recsys.train.aurora_client import AuroraConfig

from recsys.serve.dynamodb_client import DynamoDBClient,DynamoDBConnectionConfig

from config import config
from logger import logger

lg = logger().logger


# Nginx - Gunicorn
def sigterm_handler(nginx_pid, gunicorn_pid):
    try:
        os.kill(nginx_pid, signal.SIGQUIT)
    except OSError:
        pass
    try:
        os.kill(gunicorn_pid, signal.SIGTERM)  # gunicorn.pid
    except OSError:
        pass

    sys.exit(0)


def endpoint_server():
    cpu_count = multiprocessing.cpu_count()
    model_server_timeout = os.environ.get('MODEL_SERVER_TIMEOUT', 60)
    model_server_workers = int(os.environ.get('MODEL_SERVER_WORKERS', cpu_count))

    print("Starting the inference server with {} workers.".format(model_server_workers))

    subprocess.check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
    subprocess.check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])

    nginx = subprocess.Popen(['nginx', '-c', '/opt/program/nginx.conf'])
    gunicorn = subprocess.Popen(['gunicorn',
                                 '--timeout', str(model_server_timeout),
                                 '-k', 'gevent',
                                 '-b', 'unix:/tmp/gunicorn.sock',
                                 '-w', str(model_server_workers),
                                 'wsgi:app'])

    signal.signal(signal.SIGTERM, lambda a, b: sigterm_handler(nginx.pid, gunicorn.pid))

    pids = set([nginx.pid, gunicorn.pid])
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break

    sigterm_handler(nginx.pid, gunicorn.pid)
    print('Inference Server Exiting')

if __name__ == '__main__':
    endpoint_server()
