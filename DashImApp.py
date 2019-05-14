from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go



import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
from skimage import io, transform
py.init_notebook_mode(connected=True)

import csv


from skimage import color

from sklearn import manifold, preprocessing


import glob, os, flask

import tqdm


from numba import cuda

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import random

import os

from joblib import Parallel, delayed

#Let's setup the random seed. 
random.seed(1912)

# SET POSSIBLE DIRECTORIES TO BE LISTS OF FOLDER PATHS!!!!

# possible_directories = glob.glob('/media/leonardo/DATA/CRASSH/Animals_with_Attributes2/JPEGImages' + '/*/')

#possible_directories = glob.glob('/home/leonardo/Documents/GPUcomputing/drapery/style/humans_out' + '/*/')

#./possible_directories = glob.glob('/media/leonardo/DATA/Annunciation_Baptism/Annunciation/Others/*/')

#possible_directories = glob.glob('/home/leonardo/Documents/GPUcomputing/BEGAN-tensorflow/data/ven_bodies/')

possible_directories = glob.glob('/home/leonardo/Documents/GPUcomputing/Fototeca/Train_icon_text/*/')


#possible_directories = glob.glob('/home/leonardo/Documents/GPUcomputing/Messicano/*/')


# possible_directories = glob.glob('./imout' + '/*/')
static_image_route = '/static/'
# image_directory = '/home/leonardo/Documents/GPUcomputing/Fototeca/Train_materials/Öl/'
# image_directory = './imout/flemish/'





def makeThumbnail(imfn, baseheight):
    testim = io.imread(imfn)
    hpercent = (baseheight / float(testim.shape[1]))
    wsize = int((float(testim.shape[0]) * float(hpercent)))
    smallim = transform.resize(testim,(wsize, baseheight))
    if len(testim.shape) == 2:
        smallim = color.grey2rgb(smallim)
    io.imsave('./tmpim/' + imfn.split('/')[-1],smallim)

def prepareImages(image_directory, Nmax=50):
    for file in os.listdir('./tmpim/'):
        if file.endswith('.jpg'):
            os.remove('./tmpim/' + file)

    list_of_images = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(image_directory))]
    NIms = len(list_of_images)
    #randomly rearrange so we don't have bias by files...
    random.shuffle(list_of_images)

    if(NIms>Nmax):
        list_of_images = list_of_images[:Nmax]
        NIms = len(list_of_images)

    ImPaths = [image_directory + fn for fn in list_of_images]
    ImPaths = ImPaths[:Nmax]

    baseheight = 256.0
    ##Make the thumbnails change size in extreme cases
    if Nmax > 300:
        baseheight = 128.0
    if Nmax < 200:
        baseheight = 512.0

    print('writing thumbnails')

    Parallel(n_jobs=36)(delayed(makeThumbnail)(i, baseheight) for i in ImPaths)

    print('wrote thumbnails')

    image_directory = './tmpim/'


    ImPaths = [image_directory + imf for imf in list_of_images]

    ImURLs = [static_image_route + imf for imf in list_of_images]

    return [ImPaths, ImURLs]



model_dir = './model/'

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long




def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
    """Runs inference on an image.

    Args:
    image: Image file name.

    Returns:
    Nothing
    """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    return predictions


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)



maybe_download_and_extract()








app = dash.Dash()





# Add a static image route that serves images from desktop
# Be *very* careful here - you don't want to serve arbitrary files
# from your computer or server


@app.server.route('{}<image_path>.jpg'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.jpg'.format(image_path)
#     if image_name not in list_of_images:
#         raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory('./tmpim/', image_name)






def processImages(ImPaths, ImURLs):

    hoverText = []
    for i in range(len(ImURLs)):
        hoverText.append(  ImURLs[i].split('/')[-1].split('.jpg')[0] )

    red = []
    green = []
    blue = []
    hue = []
    bright = []
    neuralV = []

    cuda.select_device(0)

    SF = 10000.0 #Scale Factor
    AR = 2.0 ## Aspect Ratio: x / y

    # This should allow some dynamic memory allocation...
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:


        create_graph()

#         ops = sess.graph.get_operations()
#         for o in ops:
#             print(o.values())
        softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')


        for i in tqdm.tqdm(range( len(ImPaths) )):
            myim_filename = ImPaths[i]
            im = io.imread(myim_filename)
            greyim = color.rgb2gray(im)
            maskim = greyim>0.05

            avim = np.mean(im.flatten())
            red.append(   int(np.mean(im[maskim,0].flatten())  )  )
            green.append( int(np.mean(im[maskim,1].flatten())  )   )
            blue.append(  int(np.mean(im[maskim,2].flatten())  )    )
            hue.append(color.rgb2hsv([[[red[-1],green[-1],blue[-1] ]]])[0][0][0] )
            bright.append(avim)

            image_data = tf.gfile.FastGFile(myim_filename, 'rb').read()
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)
            neuralV.append(predictions)

##          I tried to make this parallel, but to no avail...
#         image_data = []
#         for i in tqdm.tqdm(range( len(ImPaths) )):
#             myim_filename = ImPaths[i]
#             im = io.imread(myim_filename)
#             image_data.append(tf.gfile.FastGFile(myim_filename, 'rb').read() )


#         all_image_data = tf.stack(image_data)
#         predictions = sess.run(softmax_tensor,
#             {'DecodeJpeg/contents:0': all_image_data})






    dataMatrix = np.asarray(neuralV)

    tsne_out = manifold.TSNE(n_components=2).fit_transform(dataMatrix).transpose()

    tsne_out = preprocessing.normalize(tsne_out)

    FirstVec = tsne_out[0]
    SecondVec = tsne_out[1]

    FirstVec *= SF * AR
    SecondVec *= SF

    mytrace = go.Scatter(
        x=FirstVec,
        y=SecondVec,
        text=hoverText,
        hoverinfo='text',
        showlegend=False,
        mode = 'markers',
    )

    return [mytrace, FirstVec, SecondVec]

# def update_point(trace, points, selector):
#     print('clicked')
# #     c = list(scatter.marker.color)
# #     s = list(scatter.marker.size)
# #     for i in points.point_inds:
# #         c[i] = '#bae2be'
# #         s[i] = 20
# #         scatter.marker.color = c
# #         scatter.marker.size = s


# trace0.on_click(update_point)



def layoutFromImages(ImURLs, FirstVec, SecondVec):
    layout = {}
    layout['images'] = []
    for i in range( len(ImURLs) ):
        imsize = 100
        if (  len(ImURLs)  ) > 300:
            imsize = 50
        if (  len(ImURLs)  ) <  200:
            imsize = 200

        layout['images'].append(dict(
            source= ImURLs[i],
            xref= "x",
            yref= "y",
            x= FirstVec[i],
            y= SecondVec[i],
            sizex= imsize,
            sizey= imsize,
            xanchor= "center",
            yanchor= "middle"  ##This really is how the two have to be defined..... seems pretty silly
          ))
    layout['hovermode'] = 'closest'
    layout['xaxis'] = dict(
        )
    layout['yaxis'] = dict(
            scaleanchor = "x"
        )
    return layout


# [ImPaths, ImURLs] = prepareImages(image_directory)

# [trace0, FirstVec, SecondVec] = processImages(ImPaths, ImURLs)

# data = [trace0]

# layout = layoutFromImages(ImURLs, FirstVec, SecondVec)


data = []
layout = {}


app.layout = html.Div([


    html.Div([

            dcc.Dropdown(
                id='directory_choice',
                options=[{'label': i.split('/')[-2], 'value': i} for i in possible_directories],
                value=possible_directories[0],
                style={'width': 400}
            ),
            dcc.Dropdown(
                id='num_images',
                options=[{'label': str(i), 'value': i} for i in [50, 100, 200, 300, 400, 500, 750, 1000]],
                value=200,
                style={'width': 400}
            )
    ]),


        dcc.Graph(
            figure=go.Figure(data,layout=layout),
            style={'height': 900},
            id='my-graph'
        )

])



@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('directory_choice', 'value'),
    dash.dependencies.Input('num_images','value')])

def update_graph(directory_choice, num_images):
    print('updated')

    [ImPaths, ImURLs] = prepareImages(directory_choice, Nmax=num_images)

    [trace0, FirstVec, SecondVec] = processImages(ImPaths, ImURLs)

    data = [trace0]

    layout = layoutFromImages(ImURLs, FirstVec, SecondVec)

    return {

        'data': data,
        'layout': layout

#         'data': [go.Scatter(
#             x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
#             y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
#             text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
#             mode='markers',
#             marker={
#                 'size': 15,
#                 'opacity': 0.5,
#                 'line': {'width': 0.5, 'color': 'white'}
#             }
#         )],
#         'layout': go.Layout(
#             xaxis={
#                 'title': xaxis_column_name,
#                 'type': 'linear' if xaxis_type == 'Linear' else 'log'
#             },
#             yaxis={
#                 'title': yaxis_column_name,
#                 'type': 'linear' if yaxis_type == 'Linear' else 'log'
#             },
#             margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
#             hovermode='closest'
#         )
    }


# cuda.close()

if __name__ == '__main__':
        app.run_server(debug=False,  host='0.0.0.0', port=8002)
