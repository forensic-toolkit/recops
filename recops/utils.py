import os
import uuid
import glob
from flask_sqlalchemy import SQLAlchemy
from urllib.parse     import urlparse
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image as keras_pre_image
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import base64
import random
import string
from io import BytesIO
from werkzeug.security import generate_password_hash, check_password_hash
import logging
import sys

storage_uri  = os.environ.get('STORAGE_URI',  f"file://{os.environ['HOME']}/.recops/data" )
database_uri = os.environ.get('DATABASE_URI', f"sqlite:///{os.environ['HOME']}/.recops/data/recops.db")
storage_path = urlparse(storage_uri).path

db = SQLAlchemy()

def stdout(*args, seperator=" ", eol="\n", spool=sys.stdout):
    spool.write(seperator.join([ f"{a}" for a in args ]) + eol)

def setup_logging(log_level="INFO", 
                    log_file=None,
                    # log_format='%(levelname)8s | %(asctime)s | %(module)s:%(pathname)s:%(lineno)d | %(message)s'
                    log_format='%(asctime)s - %(levelname)8s - %(filename)s:%(lineno)d - %(message)s'):
    level     = getattr(logging, log_level)
    formatter = logging.Formatter(log_format, datefmt='%Y/%m/%d %H:%M:%S')
    root      = logging.getLogger()
    root.setLevel(level)
    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    root.addHandler(handler)

def new_db_session():
    """
    Returns Sqlalchemy session outside of Flask.
    It is required for multiprossesing
    """
    return db.create_scoped_session()

def check_UUID(var):
    """
    Checks if given `var` is a valid UUID
    """
    try:
        uuid.UUID(var)
    except Exception as e:
        return False
    return True

def thumbgen_filename(filename):
    """
        Generate thumbnail name from filename.
    """
    name, ext = os.path.splitext(filename)
    return f'{name}_thumb{ext}'


def preprocess(img, target_size=(224, 224), grayscale=False):
    """
    
    Copy from https://github.com/serengil/deepface/blob/36ef4dc3f106f20c7c8ae3897b5a9aa7128d6099/deepface/commons/functions.py#L195
    
    """
    #post-processing
    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image
    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)
        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)
        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        if grayscale == False:
            # Put the base image in the middle of the padded image
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
        else:
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')
    #------------------------------------------
    #double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)
    #---------------------------------------------------
    #normalizing the image pixels
    img_pixels = keras_pre_image.img_to_array(img) #what this line doing? must?
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255 #normalize input in [0, 1]
    #---------------------------------------------------
    return img_pixels

def get_image_paths(target_path):
    if target_path.endswith('/'):
        target_path = target_path[:-1]
    if os.path.isdir(target_path):
        return    glob.glob(target_path + r'/*/*.jpeg', recursive=True) \
                + glob.glob(target_path + r'/*/*.jpg', recursive=True) \
                + glob.glob(target_path + r'/*/*.png', recursive=True) \
                + glob.glob(target_path + r'/*.jpeg', recursive=True) \
                + glob.glob(target_path + r'/*.jpg', recursive=True) \
                + glob.glob(target_path + r'/*.png', recursive=True)
    if os.path.isfile(target_path):
        if target_path.endswith('jpg') \
            or target_path.endswith('jpeg') \
            or target_path.endswith('png'):
            return [ target_path ]
    return []
