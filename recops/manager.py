import hashlib
import os
import copy
import numpy as np
from sqlalchemy import select
from PIL        import Image as PILImage
from io         import BytesIO

from   deepface.commons  import distance as dst

from   recops.models    import *

class RecopsManager(object):
    
    dataset = None
    session = None
    
    def __init__(self, dataset_id=None, session=None):
        self.session = session
        if dataset_id and self.session:
            self.dataset = self.session.query(Dataset).get(dataset_id)
    
    def create_image_if_not_exists(self, raw_image_data:bytes, content_type:str="", commit=True, consent=False, mark="DETECT")->tuple:
        """
        Create image object from raw data if not exists otherwise return existing 
        returns: [bool, object]
            bool: True if new or False if already exists
            object: Image object or None if failed
        """
        # Compute image file checksum
        file_checksum = hashlib.md5(raw_image_data).hexdigest()
        # Check if image already exists in our database
        img = self.session.query(Image).filter(
                                            Image.dataset_id==self.dataset.id, 
                                            Image.checksum==file_checksum).first()
        # Return existing image
        if img:
            return False, img
        # Otherwise create it
        else:
            # Try to parse image with Pillow
            img_data = None
            try:
                img_data = PILImage.open(BytesIO(raw_image_data))
            # If there is any problem return error.
            except Exception as e:
                logging.error(f"We got exception while creating image with checksum {file_checksum}")
                # raise e
                return False, None
            # If content type is not defined get it from image.
            # Generally we should allow Pillow to detect mimetype. 
            if not content_type:
                content_type = img_data.get_format_mimetype()
            # Populate image object
            img              = Image()
            img.path         = f"image/{file_checksum}"
            img.checksum     = file_checksum
            img.content_type = content_type
            img.dataset_id   = self.dataset.id
            img.consent      = consent
            img.marked       = mark
            # Save actual image file if file doesn't exist already
            if not os.path.exists(img.full_path):
                # Create sub folder if not exists
                os.makedirs(os.path.sep.join(img.full_path.split(os.path.sep)[:-1]), exist_ok=True)
                # Save new image to path
                with open(img.full_path, 'wb') as f:
                    f.write(raw_image_data)
            # Save thumbnail if doesn't exists
            thumb_path = thumbgen_filename(img.full_path)
            if not os.path.exists(thumb_path):
                # Create thumbnail image
                thumb = img_data.copy()
                thumb.thumbnail((100, 100), PILImage.ANTIALIAS)
                if thumb.mode not in ('RGB', 'RGBA'):
                    thumb = thumb.convert('RGBA')
                # Create sub folder if not exists
                os.makedirs(os.path.sep.join(thumb_path.split(os.path.sep)[:-1]), exist_ok=True)
                # Save new image to path
                try:
                    with open(thumb_path, 'wb') as f:
                        thumb.save(f, 'JPEG')
                except Exception as e:
                    logging.error(f"We got exception while creating image with checksum {file_checksum}")
                    return False, None
        if commit:
            self.session.add(img)
            self.session.commit()
        return True, img
    
    def detect_faces(self, img:Image, model=DetectedFace, align=True, commit=True, consent=None)->list:
        """
        Main function to detects faces and optionally align them (by default it aligns faces)
        based on dataset's detector model. 
        The function will check if the detected face is already stored (based on its checksum)
        otherwise it will store it. If it is new will return [ True, Face Object ] otherwise
        will return (False, Face Object). 

        returns: [ 
            ( boolean, Face Object ),
            ( boolean, Face Object ),
            ...
        ]  
        """
        if consent == None:
            consent = img.consent
        try:
            for detected_face_info in self.dataset.detector_wrapper.detect_face(
                                        self.dataset.detector,
                                        img.img_array,
                                        align=align):
                boxx       = detected_face_info[1]
                box_left   = boxx[0]
                box_top    = boxx[1]
                box_right  = boxx[0] + boxx[2]
                box_bottom = boxx[1] + boxx[3]
                # np.int64 => int
                if isinstance( box_left, np.int64) or isinstance( box_left, np.int32):
                    box_left   = box_left.item()
                    box_top    = box_top.item()
                    box_right  = box_right.item()
                    box_bottom = box_bottom.item()
                elif isinstance( box_left, int):
                    box_left   = box_left
                    box_top    = box_top
                    box_right  = box_right
                    box_bottom = box_bottom
                else:
                    print(type(box_left))
                    raise ValueError("WHAT THE FUCK ??")
                
                # Extract original face (not aligned) from image
                raw_data = img.img_array[box_top:box_bottom, box_left:box_right]
                # Compute md5 checksum of numpy array
                checksum = hashlib.md5(raw_data.tobytes()).hexdigest()
                # Check if there is already a (detected) face with that checksum
                obj = self.session.query(model).filter(model.dataset_id==self.dataset.id, model.checksum==checksum).first()
                # Otherwise create it
                if not obj:
                    obj = model()
                    obj.descriptor = self.extract_landmarks_from_detected_face_info(detected_face_info)
                    obj.dataset_id = self.dataset.id
                    obj.image_id   = img.id
                    obj.consent    = consent
                    obj.box_left   = box_left
                    obj.box_top    = box_top
                    obj.box_right  = box_right
                    obj.box_bottom = box_bottom
                    obj.checksum   = checksum
                    if commit:
                        self.session.add(obj)
                        self.session.commit()
                    yield True, obj
                else:
                    yield False, obj
        except Exception as e:
            logging.error(f"We got exception while detecting faces on {img}")
            logging.debug(str(e))


    
    def extract_landmarks_from_detected_face_info(self, detected_face_info)->list:         
        """
        Following deepface procedures extract face landmarks from a detected face.
        The function will preprocess and normalize the detected face before 
        extracting the landmarks.

        returns: landmarks
        """
        target_size    = find_input_shape(self.dataset.basemodel)
        face_img_array = preprocess(detected_face_info[0], target_size)
        # This is the default in deepface
        face_img_array = normalize_input(face_img_array, "base")
        # face_img_array = functions.normalize_input(face_img_array, img.dataset.basemodel_backend)
        descriptor = self.dataset.basemodel.predict(face_img_array)
        return descriptor[0].tolist()
    
    def compute_distances_face(self, target_face_descriptor, threshold:float=None, dataset_faces:list=None):
        """
        Iterates through `dataset_faces` (by default all faces on the dataset),
        compares distance against threshold (by default using the deepface one),
        and returns list of faces closest to the threshold 
        returns: [ 
            ( <threshold>, <distance:float>, <face object> )
        ]
        """
        if not threshold:
            if self.dataset.threshold:
                threshold = self.dataset.threshold
            else:
                threshold = self.dataset.default_threshold

        if dataset_faces == None:
            dataset_faces = self.dataset.faces
        for face in dataset_faces:
            distance = self.dataset.distance(target_face_descriptor, face.descriptor) 
            if bool( distance <= threshold ):
                yield ( threshold, distance.item(), face )

    def recognize_face(self, detected_face, threshold:float=None, dataset_faces:list=None, commit=True):
        if dataset_faces == None:
            dataset_faces = self.dataset.faces
        assert detected_face.__class__.__name__ == "DetectedFace"
        for threshold, distance, face in self.compute_distances_face(
                                                detected_face.descriptor,
                                                    dataset_faces=dataset_faces,
                                                    threshold=threshold):
            assoc = db.session.query(MatchedFaceAssociation).filter(
                            MatchedFaceAssociation.face_id==face.id,
                            MatchedFaceAssociation.detected_face_id==detected_face.id,
                        ).first()
            if not assoc:
                assoc = MatchedFaceAssociation()
                assoc.distance  = distance
                assoc.threshold = threshold
                assoc.face_id   = face.id
                assoc.detected_face_id = detected_face.id
                if commit:
                    self.session.add(assoc)
                    self.session.commit()
                yield True, assoc
            else:
                yield False, assoc