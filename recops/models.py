import os
import sys
import uuid
import numpy as np
import hashlib
import copy
import json
from base64 import b64encode
import PIL
import io

from sqlalchemy.ext.declarative     import declared_attr
from sqlalchemy.types               import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.event               import listens_for # Used for file upload
from sqlalchemy.ext.hybrid          import hybrid_property
from sqlalchemy.orm                 import join, sessionmaker, scoped_session, reconstructor, validates

from recops.utils import *

import deepface.detectors
import deepface.basemodels
from deepface.commons.functions import find_input_shape, normalize_input
from deepface.commons           import distance as dst


"""

CUSTOM DATA TYPES FOR SQLITE 
============================

Once moved to Postgres they are not gonna need anymore

"""
class GUID(TypeDecorator):
    """Platform-independent GUID type.

    Uses PostgreSQL's UUID type, otherwise uses
    CHAR(32), storing as stringified hex values.

    See: https://docs.sqlalchemy.org/en/14/core/custom_types.html#backend-agnostic-guid-type
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
    #     if dialect.name == 'postgresql':
    #         return dialect.type_descriptor(UUID())
    #     else:
            return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        # elif dialect.name == 'postgresql':
        #     return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return "%.32x" % uuid.UUID(value).int
                # return uuid.UUID(value).hex
            else:
                # hexstring
                return "%.32x" % value.int
                # return value.hex

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        # elif dialect.name == 'postgresql':
        #     return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)
            return value



class UniqueObject(object):
    """
    Underline Mixin Class to use as a base for models.

    1. Uses uuid as primary key. This gives the advantage to refer to the object without
    the need of an incremental intreger which is tightly connected to SQL db.
    2. created_at auto generated at creation
    3. updated_at auto generated during update
    4. Auto-generates table name arived from class name

    """

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()
    
    def __repr__(self):
        return f"<{self.__class__.__name__}[{self.id}]>"

    id         = db.Column(GUID(),      primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())


faceidentities = db.Table('faceidentities',
    db.Column('face_id',     GUID(), db.ForeignKey('face.id'),     primary_key=True),
    db.Column('identity_id', GUID(), db.ForeignKey('identity.id'), primary_key=True)
)

class BaseImage(UniqueObject):
    """
    Extends UniqueObject and adds some functionality for Models contain Images  
    """
    @property
    def full_path(self):
        """Returns full path to file in storage
        """
        return f"{storage_path}/{self.path}"

    @property
    def content(self):
        """Returns raw file content
        """
        return open(self.full_path, 'rb').read()
    
    @property
    def img_array(self):
        """Convert file to image numpy array
        """
        # return dlib.load_rgb_image(self.full_path)
        return np.array(PIL.Image.open(self.full_path))

    @property
    def datauri(self):
        """Convert file to Data URI Scheme (used to embeed image inside SVG)
        """
        return f"data:{self.content_type};base64,{b64encode(self.content).decode()}"

    @property
    def thumb_full_path(self):
        """Returns full path to thumb file in storage
        """
        return f"{storage_path}/{thumbgen_filename(self.path)}"

    @property
    def thumb_content(self):
        """Returns raw thumb file content
        """
        return open(self.thumb_full_path, 'rb').read()
    
    @property
    def thumb_datauri(self):
        """Convert thumb file to Data URI Scheme (used to embeed thumb image inside SVG)
        """
        return f"data:image/jpeg;base64,{b64encode(self.thumb_content).decode()}"

    @property
    def width(self):
        """Return image width
        """
        return self.img_array.shape[1]

    @property
    def height(self):
        """Return image height
        """
        return self.img_array.shape[0]







class Dataset(UniqueObject, db.Model):
    """

    Dataset model is the cetral model all other models are linked to.
    Dataset holds the models being used to detect/recognize and their options.    
    
    """
    name            = db.Column(db.String, unique=True, nullable=False)
    description     = db.Column(db.String, default="")
    faces           = db.relationship("Face",         back_populates="dataset")
    detected_faces  = db.relationship("DetectedFace", back_populates="dataset")
    images          = db.relationship("Image",        back_populates="dataset")
    jobs            = db.relationship("Job",          back_populates="dataset")
    identities      = db.relationship("Identity",     back_populates="dataset")

    detector_backend      = db.Column(db.String, default="retinaface")
    basemodel_backend     = db.Column(db.String, default="ArcFace")
    
    threshold        = db.Column(db.Float(50))
    distance_metric  = db.Column(db.String, default="cosine")

    backends = dict(
        detectors=dict(
            opencv="OpenCvWrapper",
            ssd="SsdWrapper",
            dlib="DlibWrapper",
            mtcnn="MtcnnWrapper",
            retinaface="RetinaFaceWrapper",
            mediapipe="MediapipeWrapper"
        ),
        basemodels=dict(
            ArcFace='ArcFace',
            Boosting='Boosting',
            DeepID='DeepID',
            DlibResNet='DlibResNet',
            DlibWrapper='DlibWrapper',
            Facenet='Facenet',
            Facenet512='Facenet512',
            FbDeepFace='FbDeepFace',
            OpenFace='OpenFace',
            VGGFace='VGGFace',
            SFace='SFace',
        )
    )

    _detector  = None
    _basemodel = None

    @property
    def detector_wrapper(self):
        """
        returns deepface.detectors.<wrapper> object
        """
        __import__(f'deepface.detectors.{ self.backends["detectors"].get(self.detector_backend) }')
        return getattr(deepface.detectors, 
                    self.backends["detectors"].get(self.detector_backend))

    @property
    def detector(self):
        """
        returns deepface.detectors.<wrapper>.build_model(<backend>)
        """
        if not self._detector:
            self._detector = self.detector_wrapper.build_model()
        return self._detector

    @property
    def basemodel_wrapper(self):
        """
        returns deepface.detectors.<wrapper> object
        """
        __import__(f'deepface.basemodels.{ self.backends["basemodels"].get(self.basemodel_backend) }')
        return getattr(deepface.basemodels, 
                    self.backends["basemodels"].get(self.basemodel_backend))

    @property
    def basemodel(self):
        if not self._basemodel:
            if hasattr(self.basemodel_wrapper, "loadModel"):
                self._basemodel = self.basemodel_wrapper.loadModel()
            elif hasattr(self.basemodel_wrapper, "load_model"):
                self._basemodel = self.basemodel_wrapper.load_model()
            else:
                raise Exception(f'Could not load model for {self.basemodel_backend} !')
        return self._basemodel

    @property
    def face_count(self):
        return len(self.faces)

    @property
    def identity_count(self):
        return len(self.identities)
        
    @property
    def default_threshold(self):
        return dst.findThreshold(self.basemodel_backend, self.distance_metric)

    def distance(self, target_face_descriptor, source_face_descriptor):
        if self.distance_metric == "cosine":
            distance  = dst.findCosineDistance(target_face_descriptor, source_face_descriptor)
        elif self.distance_metric == 'euclidean':
            distance = dst.findEuclideanDistance(target_face_descriptor, source_face_descriptor )
        elif self.distance_metric == 'euclidean_l2':
            distance = dst.findEuclideanDistance(dst.l2_normalize(target_face_descriptor), dst.l2_normalize(source_face_descriptor))
        return np.float64(distance)
    
    def filter_faces(self, maximum=-1, identified_faces=True, exclude_identities=[]):
        """
        Filters faces based on identities.
        """
        if exclude_identities:
            logging.warning("`exclude_identities` is not in use and will not affect the results")
        if identified_faces:
            _faces = list(filter(lambda f: len(f.identities) > 0, self.faces))
        else:
            _faces = list(filter(lambda f: len(f.identities) == 0, self.faces))
        if _faces:
            # if exclude_identities:
            #     for exclude_identity in exclude_identities:
            #         _faces = list(filter(lambda f: exclude_identity not in f.identity_ids, _faces))
            if maximum < 0:
                return _faces
            else:
                return _faces[:maximum]
        return []


    def __repr__(self):
        return f"<Dataset[ id={self.id} name={self.name} detector={self.detector_backend} basemodel={self.basemodel_backend} ]>"





class Identity(UniqueObject, db.Model):
    """    
    
    Identity represents an Legal Entity that can be identified, usually a Human Being.
    
    """
    name        = db.Column(db.String, unique=True, nullable=False)
    
    # Dataset used to serialize Image
    dataset_id  = db.Column(GUID(), db.ForeignKey('dataset.id'))
    dataset     = db.relationship("Dataset", back_populates="identities")

    attributes  = db.Column(db.JSON,   default={}) 
    description = db.Column(db.Text)
    
    color       = db.Column(db.String, default="#00ff01")
    
    @property
    def face_ids(self):
        return [ str(i.id) for i in self.faces ]

    def __repr__(self):
        return f"<Identity[id={self.id} name={self.name} faces={','.join(self.face_ids)} ]>"



class Image(BaseImage, db.Model):
    """
    
    Image is the main interface to handle general images.

    Attributes

    id:            Unique UUID to reference this file. 
    source_uri:    [Optional] Original Uri where the file comes from.
    description:   [Optional] Free text form to add additional information about file
    identities:    [Optional] List of identities Image is linked to.    
    faces:         [Optional] List of faces Image is linked to.    

    full_path:     Dynamic attribute points to full path of local file 
    content:       Dynamic attribute returns image content

    """
    
    dataset_id    = db.Column(GUID(), db.ForeignKey('dataset.id'))
    dataset       = db.relationship("Dataset", back_populates="images")

    path          = db.Column(db.String)

    description   = db.Column(db.Text,    default="")
    
    jobs          = db.relationship("Job",   back_populates="image")

    # MD5 Checksum of the file 
    checksum      = db.Column(db.String)
    
    content_type  = db.Column(db.String,  default="image/jpeg")
    
    # Linked faces
    faces = db.relationship("Face", back_populates="image")

    # Detected faces
    detected_faces = db.relationship("DetectedFace", back_populates="image")

    # 
    marked  = db.Column(db.String,  default="DETECT")   # IMPORT   => Import for dataset
                                                        # DETECT   => Used for detection 
    consent = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"<Image[id={self.id} path={self.path}]>"



@listens_for(Image,   'after_delete')
def del_image(mapper, connection, target):
    """
    SQLALchemy Event listener to delete actual image files
    from storage once object is removed from the database.
    """
    if target.__tablename__ in ['image']:
        if target.path:
            # Delete file
            try:
                os.remove(target.full_path)
            except OSError:
                pass
            # Delete thumbnail for Images
            if target.__tablename__ == 'image':
                try:
                    os.remove(thumbgen_filename(target.full_path))
                except OSError:
                    pass



class Face(BaseImage, db.Model):
    """
    
    Face is an object linked to an image that holds a descriptor.
    Descriptor is a pickled field contains the actual landmarks of the face.
    Face is not linked to any local file, load its content from parent image 
    using bounded box (box_left,box_top,box_right,box_bottom).
    Checksum is computed from the pixels of bounded box so it is consistent
    and can be used to check 2 different faces.


    https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk

    for chunk in numpy.nditer(
            array, flags=['external_loop', 'buffered', 'zerosize_ok'],
            buffersize=buffersize, order='C'):
        fp.write(chunk.tobytes('C'))

    raw = io.BytesIO()
    fc.tofile(raw)
    >>> np.savetxt('test.out', x, delimiter=',')   # X is an array

    """
    # One to many relationship to dataset
    dataset_id    = db.Column(GUID(), db.ForeignKey('dataset.id'))
    dataset       = db.relationship("Dataset", back_populates="faces")
    
    # One to many relationship to identity
    # identity_id   = db.Column(GUID(), db.ForeignKey('identity.id'))
    # identity      = db.relationship("Identity", back_populates="faces")
    identities      = db.relationship("Identity",
                            secondary=faceidentities,
                            # lazy='subquery',
                            # backref=db.backref('pages', lazy=True)
                            backref="faces",
                            )
    
    # One to many relationship to image
    image_id  = db.Column(GUID(), db.ForeignKey('image.id'))
    image     = db.relationship("Image", back_populates="faces")

    # MD5 checksum of numpy array
    checksum       = db.Column(db.String)

    # Descriptor is the serialized face 
    descriptor     = db.Column(db.PickleType())
    
    # Bounding box
    box_left       = db.Column(db.Integer)
    box_top        = db.Column(db.Integer)
    box_right      = db.Column(db.Integer)
    box_bottom     = db.Column(db.Integer)
 
    # 
    matched_faces  = db.relationship("MatchedFaceAssociation", back_populates="face")

    # 
    consent        = db.Column(db.Boolean, default=False)

    # Extended fields 
    gender_prediction  = db.Column(db.PickleType())
    gender             = db.Column(db.String)
    emotion_prediction = db.Column(db.PickleType())
    emotion            = db.Column(db.String)
    age_predictions    = db.Column(db.PickleType())
    age                = db.Column(db.Integer)
    race_prediction    = db.Column(db.PickleType())
    race               = db.Column(db.String)
    
    @property
    def img_array(self):
        """
        Extract face numpy array from parent image
        """
        if self.image:
            return self.image.img_array[self.box_top:self.box_bottom, self.box_left:self.box_right]
        return None
    
    @property
    def identity_names(self):
        return [ i.name for i in self.identities ]

    @property
    def identity_ids(self):
        return [ str(i.id) for i in self.identities ]

    @property
    def datauri(self):
        """Convert file to Data URI Scheme (used to embeed image inside SVG)
        """
        return f"data:image/jpeg;base64,{b64encode(self.content).decode()}"
    
    @property
    def content(self):
        face_img_array = PIL.Image.fromarray(self.img_array)
        face_img_array_raw = io.BytesIO()
        face_img_array.save(face_img_array_raw, format='JPEG')
        return face_img_array_raw.getvalue()
        
    def __repr__(self):
        return f"<Face[ id={self.id} image={self.image_id} identities={','.join(self.identity_names)} ]>"



class MatchedFaceAssociation(db.Model):
    __tablename__ = 'matched_face_association'
    
    # left
    detected_face_id  = db.Column(GUID(), db.ForeignKey('detectedface.id'), primary_key=True)
    detected_face     = db.relationship("DetectedFace", back_populates="matched_faces")

    # right
    face_id    = db.Column(GUID(), db.ForeignKey('face.id'), primary_key=True)
    face       = db.relationship("Face", back_populates="matched_faces")
    
    distance  = db.Column(db.Float(50))
    threshold = db.Column(db.Float(50))

    @property
    def detected_face_img_array(self):
        return self.detected_face.img_array

    @property
    def face_img_array(self):
        return self.face.img_array

    @property
    def percentage(self):
        """
        Compute percentage from distance 
        """
        return self.distance 

    def __repr__(self):
        return f"""<MatchedFaceAssociation[ distance={self.distance} {self.detected_face} {self.face} ]>"""


class DetectedFace(BaseImage, db.Model):
    """
    
    """

    # One to many relationship to dataset
    dataset_id    = db.Column(GUID(), db.ForeignKey('dataset.id'))
    dataset       = db.relationship("Dataset", back_populates="detected_faces")

    # One to many relationship to image
    image_id    = db.Column(GUID(), db.ForeignKey('image.id'))
    image       = db.relationship("Image", back_populates="detected_faces")

    # MD5 checksum of numpy array
    checksum       = db.Column(db.String)

    # Descriptor is the serialized face
    descriptor  = db.Column(db.PickleType())

    # Bounding box
    box_left    = db.Column(db.Integer)
    box_top     = db.Column(db.Integer)
    box_right   = db.Column(db.Integer)
    box_bottom  = db.Column(db.Integer)

    # List of faces
    matched_faces  = db.relationship("MatchedFaceAssociation")

    consent           = db.Column(db.Boolean, default=False)

    # Extended fields 
    gender_prediction  = db.Column(db.PickleType())
    gender             = db.Column(db.String)
    emotion_prediction = db.Column(db.PickleType())
    emotion            = db.Column(db.String)
    age_predictions    = db.Column(db.PickleType())
    age                = db.Column(db.Integer)
    race_prediction    = db.Column(db.PickleType())
    race               = db.Column(db.String)

    @property
    def img_array(self):
        """
        Extract face numpy array from parent image
        """
        if self.image:
            return self.image.img_array[self.box_top:self.box_bottom, self.box_left:self.box_right]
        return None

    @property
    def datauri(self):
        """Convert file to Data URI Scheme (used to embeed image inside SVG)
        """
        return f"data:image/jpeg;base64,{b64encode(self.content).decode()}"
    
    @property
    def content(self):
        face_img_array = PIL.Image.fromarray(self.img_array)
        face_img_array_raw = io.BytesIO()
        face_img_array.save(face_img_array_raw, format='JPEG')
        return face_img_array_raw.getvalue()

    @property
    def has_matched_faces(self):
        return len(self.matched_faces) > 0

    def filter_matched_faces(self, maximum=-1, identified_faces=True, exclude_identities=[], threshold=None):
        """
        Filters matched faces and sorts results by distance.
        """
        if exclude_identities:
            logging.warning("`exclude_identities` is not in use and will not affect the results")
        _matched_faces = self.matched_faces
        if identified_faces:
            _matched_faces = list(filter(lambda f: len(f.face.identities) > 0, self.matched_faces))
        if threshold != None:
            _matched_faces = list(filter(lambda f: f.distance <= threshold, _matched_faces))
        if _matched_faces:
            if maximum < 0:
                return sorted(_matched_faces, key=lambda f: f.distance)
            else:
                return sorted(_matched_faces, key=lambda f: f.distance)[:maximum]
        return []
    
    def best_identities(self, threshold=None):
        """
        Returns the best matched identity or none
        """
        for matched_face in self.filter_matched_faces(maximum=1, identified_faces=True, threshold=threshold):
            return matched_face.face.identities
        return []

    def color(self, threshold=None):
        color = None
        if not self.has_matched_faces:
            color = "#ff5702" # Irrelevant
        else:
            for idn in self.best_identities(threshold=threshold):
                color = idn.color
            if not color:
                color = "#fea805"
        return color

    @property
    def short_id(self):
        return str(self.id)[:8]

    def __repr__(self):
        return f"<DetectedFace[ id={self.id} ]>"




class Job(UniqueObject, db.Model):
    """    
    
    """
    dataset_id  = db.Column(GUID(), db.ForeignKey('dataset.id'))
    dataset     = db.relationship("Dataset", back_populates="jobs")

    image_id    = db.Column(GUID(), db.ForeignKey('image.id'))
    image       = db.relationship("Image", back_populates="jobs")

    processed   = db.Column(db.Boolean, default=False) 

    options     = db.Column(db.JSON,   default={}) 
    
    def __repr__(self):
        return f"<Job[id={self.id} image_id={self.image_id}]>"
