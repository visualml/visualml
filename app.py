#!/usr/bin/python
# -*- coding: utf-8 -*-
from aiohttp import web
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential, load_model
from keras.callbacks import Callback
from keras.models import model_from_json
import os
import time
import json
import asyncio
import socketio
import threading
import numpy as np
import pandas as pd
import PIL
from PIL import Image


#### CONSTANTS & STUFF ####

def run_nested_until_complete(future, loop=None):
    if loop is None:
        loop = asyncio.get_event_loop()

    loop._check_closed()
    if not loop.is_running():
        raise RuntimeError('Event loop is not running.')
    new_task = not isinstance(future, asyncio.futures.Future)
    task = asyncio.tasks.ensure_future(future, loop=loop)
    if new_task:

        # An exception is raised if the future didn't complete, so there
        # is no need to log the "destroy pending task" message

        task._log_destroy_pending = False
    while not task.done():
        try:
            loop._run_once()
        except:
            if new_task and future.done() and not future.cancelled():

                # The coroutine raised a BaseException. Consume the exception
                # to not log a warning, the caller doesn't have access to the
                # local task.

                future.exception()
            raise
    return task.result()


class LossHistory(Callback):

    def __init__(self, sid):
        self.sid = sid
        super().__init__()

    def on_batch_end(self, batch, logs={}):
        run_nested_until_complete(sio.emit('loss',
                                  data=str(logs.get('loss')),
                                  room=self.sid))


DATASET_INFO = {'mnist': {'input_size': (28, 28, 1),
                'output_size': 10}, 'cifar': {'input_size': (32, 32,
                3), 'output_size': 10}}

CIFAR_DICT = [
    'Frog',
    'Truck',
    'Deer',
    'Automobile',
    'Bird',
    'Horse',
    'Ship',
    'Cat',
    'Airplane',
    'Dog'
]

#### SETUP ####

sio = socketio.AsyncServer(async_mode='aiohttp', ping_timeout=3600)
app = web.Application()
sio.attach(app)

users = {}


async def index(request):
    with open('app.html') as f:
        return web.Response(text=f.read(), content_type='text/html')


@sio.on('connect')
async def connect(sid, environ):
    pass


@sio.on('train')
async def train(sid, data):
    layers = data['layers']
    info = data['info']
    input_shape = DATASET_INFO[info['dataset']]['input_size']
    output_shape = DATASET_INFO[info['dataset']]['output_size']

    directory = 'data/' + info['dataset']
    x = np.load(directory + '/x.npy')
    y = np.load(directory + '/y.npy')

    loss_log = LossHistory(sid)

    model = Sequential()

    for index, layer in enumerate(layers):
        if layer['type'] == 'nn':
            if index == 0:
                model.add(Dense(layer['size'],
                          activation=layer['activation'],
                          input_shape=input_shape))
            else:
                model.add(Dense(layer['size'],
                          activation=layer['activation']))
        elif layer['type'] == 'conv':
            if index == 0:
                model.add(Conv2D(layer['size'], (layer['filter_size'],
                          layer['filter_size']),
                          activation=layer['activation'],
                          input_shape=input_shape))
            else:
                model.add(Conv2D(layer['size'], (layer['filter_size'],
                          layer['filter_size']),
                          activation=layer['activation']))
        elif layer['type'] == 'pool':
            if index == 0:
                model.add(MaxPool2D(pool_size=(layer['pool_size'],
                          layer['pool_size']), input_shape=input_shape))
            else:
                model.add(MaxPool2D(pool_size=(layer['pool_size'],
                          layer['pool_size'])))
        elif layer['type'] == 'flatten':
            if index == 0:
                model.add(Flatten(input_shape=input_shape))
            else:
                model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    loss = ('sparse_' if info['dataset'] == 'cifar' else '') + 'categorical_crossentropy'
    model.compile(loss=loss, optimizer='adam')
    model.fit(x, y, epochs=info['epochs'], callbacks=[loss_log],
              batch_size=info['batch_size'])

    idd = str(int(time.time()))

    model.save('models/' + idd + '.h5')
    with open('models/' + idd + '.h5', 'rb') as saved_model:
        await sio.emit('id', data=idd, room=sid)

    with open('models/' + idd + '.json', 'w') as json_file:
        json.dump(data, json_file)

    users[sid] = model


@sio.on('run with image')
async def run_with_image(sid, data):
    idd = str(int(time.time()))

    with open('tmp/' + idd + '.image', 'wb') as outfile:
        outfile.write(data)

    img = Image.open('tmp/' + idd + '.image')
    img = img.resize((28, 28), PIL.Image.ANTIALIAS)
    img = np.divide(np.reshape(np.array(img), (1, 28, 28, 3)), 256)
    img = np.sum(img, axis=-1, keepdims=True) / 3

    model = users[sid]

    predictions = model.predict(img)[0]
    best_preds = np.argsort(predictions)

    os.remove('tmp/' + idd + '.image')

    await sio.emit('run result', data=str(best_preds[0]) + " " + str(best_preds[1]) + " " + str(best_preds[2]), room=sid)


@sio.on('from id')
async def from_id(sid, data):
    data = data[1:]
    model = load_model('models/' + data + '.h5')
    with open('models/' + data + '.json', 'r') as json_file:
        await sio.emit('model from id', data=json.load(json_file),
                       room=sid)
    users[sid] = model


@sio.on('disconnect')
async def disconnect(sid):
    if sid in users:
        del users[sid]

app.router.add_static('/static', 'static')
app.router.add_get('/', index)

web.run_app(app, port=8081)
