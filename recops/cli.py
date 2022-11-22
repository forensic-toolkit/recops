"""
Main command line interface to interact with recops
"""
import os
import click
import hashlib
import logging
import sys
import zipfile
import glob

# Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, url_for, render_template, render_template_string
from sqlalchemy.sql import text
from sqlalchemy.orm import load_only

import pickle

from recops.utils        import *
from recops.models       import *
from recops.manager      import *
from recops.app          import *

def get_object_from_UUID(model, _uuid, session):
    if check_UUID(_uuid):
        return session.query(model).filter(model.id == _uuid).first()
    return None

def get_dataset_from_UUID(dataset_id):
    ds = get_object_from_UUID(Dataset, dataset_id, db.session)
    if not ds:
        logging.error(f"Dataset with id {dataset_id} doesn't exists")
        sys.exit(1)
    return ds

@click.group()
@click.option("--log-level",   default="ERROR", help="Specify logging level")
@click.option("--log-file",    default=None, help="Specify file to output logs (by default stderr)")
@click.pass_context
def cli(ctx, log_level, log_file):
    setup_logging(log_level, log_file)
    pass

@cli.command()
@click.pass_context
def debug(ctx):
    import code
    code.interact(local=locals())

@cli.command()
@click.pass_context
def dataset_list(ctx):
    """
    List available datasets 
    """
    for ds in db.session.query(Dataset).all():
        stdout(f"""<Dataset[ 
    id: {ds.id}
    name: {ds.name}
    detector: {ds.detector_backend}
    basemodel: {ds.basemodel_backend}
    identities: {ds.identity_count}
    faces: {ds.face_count} 
    threshold: {ds.threshold}
    default_threshold: {ds.default_threshold}
    distance_metric: {ds.distance_metric}
]>""")

@cli.command()
@click.argument('name')
@click.option("--detector",    default="retinaface", help="Model to use for Face Detection, use dataset-create --help to list available choises (default: retinaface)")
@click.option("--basemodel",   default="ArcFace",    help="Model to use for Face Recognition, use dataset-create --help to list available choises (default: ArcFace)")
@click.option("--description", default="",           help="")
@click.option('--threshold',        default=None,      type=click.FLOAT,   help="Maximum distance to be used when comparing faces (unless specified the default threshold of basemodel is used)")
@click.option('--distance-metric',  default="cosine",  help="Metric to compute distance when comparing faces (cosine|euclidean|euclidean_l2)")
@click.pass_context
def dataset_create(ctx, name, detector, basemodel, description, threshold, distance_metric):    
    """
    Create dataset

    If dataset exists updates it; keep in mind `detector` and `basemodel` can't be updated
    \b
    --detector options:
            opencv
            ssd
            dlib
            mtcnn
            retinaface
            mediapipe
    \b
    --basemodel options:
            ArcFace
            Boosting
            DeepID
            DlibResNet
            DlibWrapper
            Facenet
            Facenet512
            FbDeepFace
            OpenFace
            VGGFace
    
    """
    dataset = db.session.query(Dataset).filter(Dataset.name==name).first()
    if dataset:
        verb = "Updated"
        logging.warning("Dataset with name: {name} already exists, try to update")
        if dataset.detector_backend != detector:
            logging.warning(f"Can't change detector from {dataset.detector_backend} to {detector}. Please create new dataset")
        if dataset.basemodel_backend != basemodel:
            logging.warning(f"Can't change basemodel from {dataset.basemodel_backend} to {basemodel}. Please create new dataset")
        dataset.distance_metric = distance_metric
    else:
        verb = "Created"
        dataset = Dataset()
        dataset.name = name
        dataset.description = description
        dataset.detector_backend = detector
        dataset.basemodel_backend = basemodel
    if not threshold:
        dataset.threshold = dataset.default_threshold
    else:
        dataset.threshold = threshold
    db.session.add(dataset)
    db.session.commit()
    stdout(f"{verb} dataset with ID: {dataset.id} Name: {dataset.name}")

@cli.command()
@click.option("-d", "--dataset-id", required=True,   help="Specify dataset id")
@click.pass_context
def dataset_list_images(ctx, dataset_id):
    """
    List available images in dataset
    """
    ds = get_dataset_from_UUID(dataset_id)
    for img in ds.images:
        stdout(img)

@cli.command()
@click.option("-d", "--dataset-id", required=True,   help="Specify dataset id")
@click.pass_context
def dataset_list_faces(ctx, dataset_id):
    """
    List available faces in dataset
    """
    ds = get_dataset_from_UUID(dataset_id)
    for face in ds.faces:
        stdout(face)

@cli.command()
@click.option("-d", "--dataset-id", required=True,   help="Specify dataset id")
@click.pass_context
def dataset_list_identities(ctx, dataset_id):
    """
    List available identities in dataset
    """
    ds = get_dataset_from_UUID(dataset_id)
    for identity in ds.identities:
        stdout(identity)

@cli.command()
@click.argument('dataset_id')
# @click.option("-d", "--dataset-id", required=True,   help="Specify dataset id")
@click.pass_context
def dataset_delete(ctx, dataset_id):
    """
    Delete specified dataset and all faces, images and identities linked to it
    """

    ds = get_dataset_from_UUID(dataset_id)
    # First delete DetectedFace and MatchedFaceAssociation, as MatchedFaceAssociations are
    # not directly linked to dataset.
    for d in db.session.query(DetectedFace)\
                        .filter(DetectedFace.dataset_id==ds.id)\
                        .all():
        for m in d.matched_faces:
            db.session.delete(m)
        db.session.delete(d)
        db.session.commit()
    # Then we delete models that are directly linked to dataset
    for model in [ Job, Identity, Face, Image ]:
        for obj in db.session.query(model).filter(model.dataset_id==ds.id).all():
            db.session.delete(obj)
        db.session.commit()
    db.session.delete(ds)
    db.session.commit()

@cli.command()
@click.argument('name')
@click.option("-d", "--dataset-id", required=True,   help="Specify dataset id")
@click.option('--color',       default="#00ff01", help="")
@click.option('--description', default="", help="")
@click.pass_context
def identity_create(ctx, dataset_id, name, color, description):
    """
    Create an identity
    """
    ds = get_dataset_from_UUID(dataset_id)
    if db.session.query(Identity).filter(Identity.name==name, Identity.dataset_id==ds.id).first():
        logging.warning("Identity with name: {name} already exists")
    else:
        idn             = Identity()
        idn.name        = name
        idn.color       = color
        idn.description = description
        idn.dataset_id  = ds.id
        db.session.add(idn)
        db.session.commit()
        logging.debug("Created Identity with ID: {idn.id} Name: {idn.name}")


@cli.command()
@click.argument('identity_id')
@click.argument('face_id')
@click.pass_context
def identity_link(ctx, identity_id, face_id):
    """
    Link existing identity to a face  
    """
    idn  = db.session.query(Identity).filter(Identity.id==identity_id).first()
    face = db.session.query(Face).filter(Face.id==face_id).first()
    if not face:
        logging.error("Face with id:{face_id} doesn't exist")
        sys.exit(1)
    if not idn:
        logging.error("Identity with id:{identity_id} doesn't exist")
        sys.exit(1)
    if face.dataset_id != idn.dataset_id:
        logging.error("We can link Faces with Identities only when they are part of the same dataset")
        logging.error("Identity {idn.id} is part of {idn.dataset} ")
        logging.error("Face {face.id} is part of {face.dataset} ")
        sys.exit(1)
    else:
        if face in idn.faces:
            logging.debug("Face {face.id} is linked to {idn.id} already")
        else:
            idn.faces.append(face)
            db.session.add(idn)
            db.session.commit()

@cli.command()
@click.argument('identity_id')
@click.pass_context
def identity_delete(ctx, identity_id):
    """
    Delete an identity
    """
    idn = db.session.query(Identity).filter(Identity.id==identity_id).first()
    if not idn:
        logging.error(f"Identity with id: {identity_id} does not exist")
    else:
        db.session.delete(idn)
        db.session.commit()

@cli.command()
@click.argument('target_path', type=click.Path(exists=True))
@click.option("-d", "--dataset-id", required=True,   help="Specify dataset id")
@click.option('--consent',          default=False,   help="Mark imported images/faces as have been consent to use")
@click.option('--align/--no-align', default=True,    help="Align detected faces")
@click.option('--force',            is_flag=True,    help="Force to reprocess existing objects")
@click.option('--output-objects',   default=None,    help="File path to write list of objects created (by default it doesnt output anything)")
@click.option('--output-format',    default="csv",   help="Output format for --output-objects (default: csv)")
@click.pass_context
def dataset_import_images(ctx, target_path, dataset_id, consent, align, force, output_objects, output_format):
    """
    Import images from local folder

    \b
    This is a generic import. The function will iterate through files in given folders and
    pick those with jpg, jpeg, png extentions. For each image will extract all faces (if any)
    and store them. This process will not link the faces to any identity, you should do it manually.
    If you have images grouped by identity you should consider using either dataset-import-faces or
    dataset-import-identities functions.

    \b
    Use Ctrl-C to stop at any time, changes are written to database during the process.
    Safe to rerun it over and over again.

    \b
    Example folder structure:
        <local_folder>/
        <local_folder>/image-001.jpeg
        <local_folder>/whatever-name.jpeg
        <local_folder>/unrelated.pdf      <= will skip files that are no images
        <local_folder>/unrelated.mp4      <= will skip files that are no images
        <local_folder>/another-image.png
        ...

    """
    if output_objects:
        if output_format != "csv":
            logging.error(f"Only csv is supported")
            sys.exit(1)
        output_objects = open(output_objects, 'a')
    ds = get_dataset_from_UUID(dataset_id)
    rmgr = RecopsManager(dataset_id=ds.id, session=db.session)
    with click.progressbar( get_image_paths(target_path), label=f" + Importing images ...") as filepaths:
        for filepath in filepaths:
            created, img = rmgr.create_image_if_not_exists(open(filepath, 'rb').read(), consent=consent, mark="IMPORT")
            if img == None:
                logging.error(f"Couldn't read file from {filepath}, propably wrong format")
                continue
            if created or force:
                if output_objects:
                    output_objects.write(f"Image,{img.id},{filepath}\n")
                logging.debug(f"Processing {img} from {filepath}")
                for created, face in rmgr.detect_faces(img, model=Face, align=align, consent=consent):
                    if created or force:
                        if output_objects:
                            output_objects.write(f"Face,{face.id},{filepath}\n")
                        logging.debug(f"Face {face} created from filepath:{filepath}")
            else:
                logging.debug(f"Skipping {img} from {filepath} is already processed")

@cli.command()
@click.argument('target_path', type=click.Path(exists=True))
@click.option("-d", "--dataset-id", required=True,   help="Specify dataset id")
@click.option('--consent',          default=False,   help="Mark imported images/faces as have been consent to use")
@click.option('--identity-id',      default="",      help="link faces to specified identity (None by default)")
@click.option('--align/--no-align', default=True,    help="Align detected faces")
@click.option('--force',            is_flag=True,    help="Force to reprocess existing objects")
@click.option('--output-errors',    default="",      help="Specify path to export files contain errors (by default it doesn't export anything)")
@click.option('--output-objects',   default=None,    help="File path to write list of objects created (by default it doesnt output anything)")
@click.option('--output-format',    default="csv",   help="Output format (default: csv)")
@click.pass_context
def dataset_import_faces(ctx, target_path, dataset_id, consent, identity_id, align, force, output_errors, output_objects, output_format):
    """
    Import faces from local folder

    \b
    Consider using this function when you already have a list images contain cropped faces.
    Each image should contain a single face, if no face or more than one face detected then
    the process will print an error and will not import it (to see errors set --log-level to INFO). 
    In case you want to export faces with errors use --output-errors to specify a folder 
    where error images will be copied.

    \b
    If all faces are part of the same identity then you can use --identity-id 
    and all faces will be linked to that identity (use identity-create to create an identity first)

    \b
    Use Ctrl-C to stop at any time, changes are written to database during the process.
    Safe to rerun it over and over again.

    \b
    Example folder structure:
        <local_folder>/
        <local_folder>/face-001.jpeg
        <local_folder>/face-002.jpeg
        <local_folder>/whatever-name.jpeg
        <local_folder>/blah.png
        ...
    """

    if output_errors:
        if not os.path.isdir(output_errors):
            logging.error(f"No folder exists under {output_errors}")
            sys.exit(1)
        if output_errors.endswith(os.path.sep):
           output_errors = output_errors[:-1]
    
    if output_objects:
        if output_format != "csv":
            logging.error(f"Only csv is supported")
            sys.exit(1)
        output_objects = open(output_objects, 'a')
    ds = get_dataset_from_UUID(dataset_id)
    rmgr = RecopsManager(dataset_id=ds.id, session=db.session)
    if identity_id:
        identity = db.session.query(Identity).filter(Identity.dataset_id==dataset_id, Identity.id==identity_id).first()
    else:
        identity = None

    with click.progressbar( get_image_paths(target_path), label=f" + Importing faces ...") as filepaths:
        for filepath in filepaths:

            created, img = rmgr.create_image_if_not_exists(open(filepath, 'rb').read(), consent=consent, mark="IMPORT")
            if img == None:
                logging.error(f"Couldn't read file from {filepath}, propably wrong format")
                continue
            if created or force:
                logging.debug(f"Processing {img} from {filepath}")
                faces = list(rmgr.detect_faces(img, model=Face, align=align, consent=consent))
                if len(faces) != 1:
                    logging.warning(f"filepath:{filepath} contains too many faces {len(faces)}")
                    # delete created image and faces.
                    for created, face in faces:
                        if created:
                            db.session.delete(face)
                    db.session.delete(img)
                    db.session.commit()

                    # Export file to error path
                    if output_errors:
                        error_path = os.path.sep.join(output_errors.split(os.path.sep) + filepath.split(os.path.sep)[-1:])
                        with open(error_path, 'wb') as f:
                            f.write(open(filepath, 'rb').read())

                else:
                    if output_objects:
                        output_objects.write(f"Image,{img.id},{filepath}\n")
                    created, face = faces[0]
                    if created or force:
                        if output_objects:
                            output_objects.write(f"Face,{face.id},{filepath}\n")
                        if identity:
                            if identity not in face.identities:
                                face.identities.append(identity)    
                        db.session.add(face)
                        db.session.commit()
                        logging.debug(f"Face {face} created from filepath:{filepath}")
            else:
                logging.debug(f"Skipping {img} from {filepath} is already processed")

@cli.command()
@click.argument('target_path', type=click.Path(exists=True))
@click.option("-d", "--dataset-id", required=True,   help="Specify dataset id")
@click.option('--consent',          default=False,   help="Mark imported images/faces as have been consent to use")
@click.option('--align/--no-align', default=True,    help="Align detected faces")
@click.option('--force',            is_flag=True,    help="Force to reprocess existing objects")
@click.option('--output-errors',    default="",      help="Specify path to export files contain errors (by default it doesn't export anything)")
@click.option('--output-objects',   default=None,    help="File path to write list of objects created (by default it doesnt output anything)")
@click.option('--output-format',    default="csv",   help="Output format (default: csv)")
@click.pass_context
def dataset_import_identities(ctx, dataset_id, target_path, consent, align, force, output_errors, output_objects, output_format):
    """
    Import faces linked to identities from local folder

    \b
    Consider using this function when you already have a list images contain cropped faces
    and grouped in folders named as face's identity.
    Each image should contain a single face, if no face or more than one face detected then
    the process will print an error and will not import it (to see errors set --log-level to INFO). 
    In case you want to export identities/faces with errors use --output-errors to specify a folder 
    where error images will be copied.

    \b
    Use Ctrl-C to stop at any time, changes are written to database during the process.
    Safe to rerun it over and over again.

    \b
    Folder structure should be in the following format:
        <local_folder>/
        <local_folder>/identity-name-001/
        <local_folder>/identity-name-001/face-001.jpeg
        <local_folder>/identity-name-001/face-002.jpeg
        <local_folder>/identity-name-001/face-003.jpeg
        <local_folder>/identity-name-002/
        <local_folder>/identity-name-002/face-001.jpeg
        ...

    """
    if output_errors:
        if not os.path.isdir(output_errors):
            logging.error(f"No folder exists under {output_errors}")
            sys.exit(1)
        if output_errors.endswith(os.path.sep):
           output_errors = output_errors[:-1]
    if output_objects:
        if output_format != "csv":
            logging.error(f"Only csv is supported")
            sys.exit(1)
        output_objects = open(output_objects, 'a')

    ds = get_dataset_from_UUID(dataset_id)
    rmgr = RecopsManager(dataset_id=ds.id, session=db.session)

    with click.progressbar( get_image_paths(target_path), label=f" ~ Importing faces/identities ...") as filepaths:
        for filepath in filepaths:
            created, img = rmgr.create_image_if_not_exists(open(filepath, 'rb').read(), consent=consent, mark="IMPORT")
            if img == None:
                logging.error(f"Couldn't read file from {filepath}, propably wrong format")
                continue
            if created or force:
                logging.debug(f"Processing {img} from {filepath}")
                faces = list(rmgr.detect_faces(img, model=Face, align=align, consent=consent))
                if len(faces) != 1:
                    logging.warning(f"filepath:{filepath} contains too many faces {len(faces)}")
                    # delete created image and faces.
                    for created, face in faces:
                        if created:
                            db.session.delete(face)
                    db.session.delete(img)
                    db.session.commit()

                    # Export file to error path
                    if output_errors:
                        idn_name = filepath.split(os.path.sep)[-2]
                        error_path = os.path.sep.join(output_errors.split(os.path.sep) + filepath.split(os.path.sep)[-2:])
                        os.makedirs(os.path.sep.join(error_path.split(os.path.sep)[:-1]), exist_ok=True)
                        with open(error_path, 'wb') as f:
                            f.write(open(filepath, 'rb').read())

                else:
                    if output_objects:
                        output_objects.write(f"Image,{img.id},{filepath}\n")
                    created, face = faces[0]
                    if created or force:
                        if output_objects:
                            output_objects.write(f"Face,{face.id},{filepath}\n")
                        # Create identity
                        idn_name = filepath.split(os.path.sep)[-2]
                        idn = db.session.query(Identity).filter(Identity.name==idn_name).first()
                        if not idn:
                            idn = Identity()
                            idn.name = str(idn_name)
                            idn.dataset_id = dataset_id
                            db.session.add(idn)
                            db.session.commit()
                            if output_objects:
                                output_objects.write(f"Identity,{idn.id},{filepath}\n")
                        # link face to identity
                        if idn not in face.identities:
                            face.identities.append(idn)
                        db.session.add(face)
                        db.session.commit()
                        logging.debug(f"Face {face} created from filepath:{filepath}")
            else:
                logging.debug(f"Skipping {img} from {filepath} is already processed")

@cli.command()
@click.argument('target_path', type=click.Path(exists=True))
@click.pass_context
def dataset_link_matched_faces(ctx, target_path):
    """
    Link faces to identities from given csv
    
    \b
    This is a helpful function to link faces to identities from given csv 
    compatible with export from dataset-matched-faces function.

    \b
    CSV should have at least following 2 rows: "<face id>,<identity id>\n"

    """
    with open(target_path, 'r') as file:
        for line in file.readlines():
            if line.strip():
                face_id, identity_id = line.split(',')[:2]
                face = db.session.query(Face).get(face_id)
                idn  = db.session.query(Identity).get(identity_id)
                if not idn in face.identities:
                    face.identities.append(idn)
                    db.session.add(face)
                    db.session.commit()
                    logging.debug(f"Linked {face} with {idn.name}")

@cli.command()
@click.option("-d", "--dataset-id",        required=True,     help="Specify dataset id")
@click.option('-o', '--output',            required=True,     type=click.Path(exists=False), help="Filename to export results (both csv and html)")
@click.option('--threshold',               default=None,      type=click.FLOAT,   help="Maximum distance to be used when comparing faces (unless specified the default threshold of basemodel is used)")
@click.option('--web-uri',                 default="http://127.0.0.1:5000", help="Web uri to use when formating html")
@click.option('--exclude-identities',      default="",        help="List of identities ids to exclude. Faces linked to these identities will not be included")
@click.pass_context
def dataset_match_faces(ctx, dataset_id, output, threshold, web_uri, exclude_identities):
    """
    
    Compare faces linked to identity with faces without identity for given dataset

    \b
    In case you have unknown faces in your dataset and want to find if they match with other
    identified faces this function might help. It will loop through faces without identity compare 
    them with those having an identity and export matched faces below --threshhold.

    \b
    Use --output to specify file to save results. There will be 2 different outputs 1 csv file and 1 html file. 
    The csv file can be used to feed dataset-link-matched-faces function.
    The html file can be used to go over it and check the faces manually, can be opened in any browser 
    but for it to visualize the faces properly you need to run webui function   

    """
    ds   = get_dataset_from_UUID(dataset_id)
    rmgr = RecopsManager(dataset_id=ds.id, session=db.session)
    
    identified_faces     = rmgr.dataset.filter_faces(identified_faces=True,  exclude_identities=exclude_identities.split(','))
    non_identified_faces = rmgr.dataset.filter_faces(identified_faces=False, exclude_identities=exclude_identities.split(','))
    html = open(output + ".html", 'w')
    csv  = open(output + ".csv", 'w')

    # Write header
    html.write("""
        <!doctype html>
        <html>
            <head>
                <style>table, th, td { border: 1px solid black; border-collapse: collapse;}</style>
            </head>
            <body>
                <table style="width:100%">
                <tr><th>""" + "</th><th>".join([
                    "Target face",
                    "Identity Name",
                    "Distance from identified face",
                    "Threshold used",
                    "Identified face",
                ]) + "</th></tr>\n")

    # Iterate through faces without identity attached
    with click.progressbar( non_identified_faces, label=f" ~ Matching faces ...") as _non_identified_faces:
        for non_identified_face in _non_identified_faces:
            # Try to find the closest identified face
            for threshold, distance, face in sorted(
                                            rmgr.compute_distances_face(
                                                non_identified_face.descriptor,
                                                    dataset_faces=identified_faces,
                                                    threshold=threshold),
                                            key=lambda r: r[1]
                                        ):
                # First iteration will hold the closest face
                html_line = [
                    # Target face
                    f"<a href=\"{web_uri}/face/?search={non_identified_face.checksum}\"><img src=\"{web_uri}/face/{non_identified_face.id}/raw\"></a>",
                    # Identity Names
                    f"<b>{','.join(face.identity_names)}</b>",
                    # Distance from identified face
                    str(distance),
                    # Threshold used
                    str(threshold),
                    # Identified face
                    f"<a href=\"{web_uri}/face/?search={face.checksum}\"><img src=\"{web_uri}/face/{face.id}/raw\"></a>",
                ]
                html.write("<tr><td>" + "</td><td>".join(html_line) + "</td></tr>\n")                    
                for idn in face.identities:
                    csv.write(",".join([
                            str(non_identified_face.id),
                            str(idn.id),
                            idn.name,
                            str(distance),
                            str(threshold),
                            str(face.id),
                        ]) + "\n")
                break
    html.write("</table></body></html>")


@cli.command()
@click.option("-d", "--dataset-id",     required=True, help="Specify dataset id")
@click.option('--age/--no-age',         default=True,  help="Compute age field")
@click.option('--gender/--no-gender',   default=True,  help="Compute gender field")
@click.option('--emotion/--no-emotion', default=True,  help="Compute emotion field")
@click.option('--race/--no-race',       default=True,  help="Compute race field")
@click.pass_context
def dataset_compute_extended_fields(ctx, dataset_id, age, gender, emotion, race):
    """
    Use extended models to compute additional fields for each face in a dataset
    
    \b
    This function uses weak and questionable models that categorize faces in a discriminative way.
    
    """
    import deepface.extendedmodels.Gender
    import deepface.extendedmodels.Age
    import deepface.extendedmodels.Emotion
    import deepface.extendedmodels.Race

    gender_model  = deepface.extendedmodels.Gender.loadModel()
    age_model     = deepface.extendedmodels.Age.loadModel()
    emotion_model = deepface.extendedmodels.Emotion.loadModel()
    race_model    = deepface.extendedmodels.Race.loadModel()

    ds   = get_dataset_from_UUID(dataset_id)
    rmgr = RecopsManager(dataset_id=ds.id, session=db.session)
    with click.progressbar( rmgr.dataset.faces, label=f" ~ Updating faces ...") as faces:
        for face in faces:
            img_224 = preprocess(face.img_array)
            # Compute gender
            if gender:
                face.gender_prediction = gender_model.predict(img_224)[0,:]
                if np.argmax(face.gender_prediction) == 0:
                    face.gender = "female"
                elif np.argmax(face.gender_prediction) == 1:
                    face.gender = "male"
            # Compute Age
            if age:
                face.age_predictions = age_model.predict(img_224)[0,:]
                face.age = int(deepface.extendedmodels.Age.findApparentAge(face.age_predictions))
            # Compute Race
            if race:
                face.race_prediction = race_model.predict(img_224)[0,:]
                # race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
                # sum_of_predictions = face.race_prediction.sum()
                # race = dict()
                # for i in range(0, len(race_labels)):
                #     race[ race_labels[i] ] = 100 * face.race_prediction[i] / face.race_prediction.sum()
                # resp_obj["dominant_race"] = race_labels[np.argmax(face.race_prediction)]
                face.race = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic'][np.argmax(face.race_prediction)]
            # Compute Emotion
            if emotion:
                # face.emotion_predictions = emotion_model.predict(img)[0,:]
                # emotion_predictions = models['emotion'].predict(img)[0,:]
                #     emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                #     img, region = functions.preprocess_face(img = img_path, target_size = (48, 48), grayscale = True, enforce_detection = enforce_detection, detector_backend = detector_backend, return_region = True)
                #     emotion_predictions = models['emotion'].predict(img)[0,:]
                #     sum_of_predictions = emotion_predictions.sum()
                #     resp_obj["emotion"] = {}
                #     for i in range(0, len(emotion_labels)):
                #         emotion_label = emotion_labels[i]
                #         emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                #         resp_obj["emotion"][emotion_label] = emotion_prediction
                #     resp_obj["dominant_emotion"] = emotion_labels[np.argmax(emotion_predictions)]
                img_48 = preprocess(face.img_array, target_size=(48, 48), grayscale=True)
                face.emotion_prediction = emotion_model.predict(img_48)[0,:]
                face.emotion = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'][np.argmax(face.emotion_prediction)]
            # Update face
            db.session.add(face)
            db.session.commit()

@cli.command()
@click.argument('target_path', type=click.Path(exists=True))
@click.option("-d", "--dataset-id", required=True,   help="Specify dataset id")
@click.option("-o", "--output",     required=True,   type=click.Path(exists=False), default="output.html", help="Path to save output report.")
@click.option('--threshold',               default=None,      type=click.FLOAT,   help="Maximum distance to be used when comparing faces (unless specified the default threshold of basemodel is used)")
@click.option('--maximum',                 default=10,        help="Maximum number of matched faces to print ( default: 10 )")
@click.option('--identified-faces',        default=False,     help="Search only through faces with identity attached (default: False)")
@click.option('--exclude-identities',      default="",        help="Comma seperated list of identity ids to exclude. Faces linked to these identities will not be used for recognition")
@click.option('--include-extended-fields', is_flag=True,      help="Whether to include or not extended fiels in the result (default: False)")
@click.option('--consent',                 default=False,     help="Mark imported images/faces as have been consent to use")
@click.option('--align/--no-align',        default=True,      help="Align detected faces")
@click.option('--force',                   is_flag=True,      help="Force to reprocess existing objects")
# @click.option('--output-format',           default="html",    help="Choose output format from html|txt (default: html)")
@click.option('--template',                default="",    help="Choose report template (default: image-view.html.j2)")
@click.pass_context
def recognize(ctx, target_path, dataset_id, output, threshold, maximum, identified_faces, exclude_identities, include_extended_fields, consent, align, force, template):
    """
    Runs face recognition on a given image and outputs report

    \b
    Face recognition is the task of making a positive identification of a face in a photo or 
    video image against a pre-existing database of faces. It begins with detection - distinguishing 
    human faces from other objects in the image - and then works on identification of those detected faces.
    
    \b
    The process will compare the similarity of two face images and provides a similarity score 
    based on the distance between two face vectors. A face vector is a high-dimensional representation 
    of face descriptors used to describe the features of a face that make it separable from other faces. 
    The face vector is unique to the network, not the face.

    """
    if os.path.exists(output) and not force:
        logging.error(f"Output file {output} already exists, please use a different path to export report")
        return

    logging.debug(f"Starting recognition on {target_path}")

    ds = get_dataset_from_UUID(dataset_id)
    logging.debug(f"Loaded dataset with id {dataset_id}")
    rmgr = RecopsManager(dataset_id=ds.id, session=db.session)
    logging.debug(f"Loaded manager, create or update image")
    created, img = rmgr.create_image_if_not_exists(open(target_path, 'rb').read(), consent=consent, mark="DETECT")
    if img == None:
        logging.error(f"Couldn't read file from {target_path}, propably wrong format")
        return
    if created or force:
        logging.debug(f"Image created")
        # Load dataset faces in memory
        if identified_faces:
            logging.debug(f"Loading identified faces from dataset")
            dataset_faces = ds.filter_faces(identified_faces=identified_faces)
        else:
            logging.debug(f"Loading faces from dataset")
            dataset_faces = ds.faces
        logging.debug(f"Dataset faces loaded")
        logging.debug(f"Detecting faces from given image")
        # Detect faces from Image
        detected_faces = rmgr.detect_faces(img, model=DetectedFace, align=align, consent=consent)
        logging.debug(f"Faces detected, looping through them")
        # Loop through detected faces
        for created, detected_face in detected_faces:
            logging.debug(f"Matching faces for {detected_face}")
            # Compute distances between detected face and dataset faces 
            # without commiting anything to database
            matches = rmgr.recognize_face(detected_face,
                                    threshold=threshold,
                                    dataset_faces=dataset_faces,
                                    commit=False)
            logging.debug(f"Faces are matched for {detected_face}, creating links")
            # Then get `maximum` faces and commit only them in the database. 
            for created, matched_assoc in sorted(matches, key=lambda f: f[1].distance)[:maximum]:
                db.session.add(matched_assoc)
                db.session.commit()
            logging.debug(f"Links created")

    logging.debug(f"Generating output")
    
    if not template:
        template = os.path.join(os.path.dirname(__file__), "templates/image-view.html.j2")

    with open(template) as t:
        with open(output, 'w') as f:
            f.write(
                render_template_string(
                    t.read(),
                    img=img,
                    maximum=maximum,
                    identified_faces=identified_faces,
                    exclude_identities=exclude_identities.split(','),
                    threshold=threshold,
                    include_extended_fields=include_extended_fields,))


@cli.command()
@click.argument('target_path_from', type=click.Path(exists=True))
@click.argument('target_path_to',   type=click.Path(exists=True))
@click.option("-d", "--dataset-id", required=True,   help="Specify dataset id")
@click.option('--threshold',               default=None,      type=click.FLOAT,   help="Maximum distance to be used when comparing faces (unless specified the default threshold of basemodel is used)")
@click.option('--consent',                 default=False,     help="Mark imported images/faces as have been consent to use")
@click.option('--align/--no-align',        default=True,      help="Align detected faces")
@click.pass_context
def verify(ctx, target_path_from, target_path_to, dataset_id, threshold, consent, align):
    """
    Runs face verification on a given image pair and outputs report
    
    \b
    This function will comparing a candidate face to another, and verifying whether it is a match. 
    It is a one-to-one mapping. This procedure will not write anything in the database, will just 
    detect the faces in the given input images, will compare the similarity of two face images and 
    will provides a similarity score based on the distance between two face vectors. The dataset is 
    used to get the detector and basemodel and has no interaction with it's linked identities or 
    faces whatsoever.   

    """
    ds   = get_dataset_from_UUID(dataset_id)
    rmgr = RecopsManager(dataset_id=ds.id, session=db.session)
    _, img_from = rmgr.create_image_if_not_exists(open(target_path_from, 'rb').read(), consent=consent, mark="DETECT", commit=False)
    if img_from == None:
        logging.error(f"Couldn't read file from {target_path_from}, propably wrong format")
        return
    _, img_to   = rmgr.create_image_if_not_exists(open(target_path_to,   'rb').read(), consent=consent, mark="DETECT", commit=False)
    if img_to == None:
        logging.error(f"Couldn't read file from {target_path_to}, propably wrong format")
        return
    detected_faces_from = list(rmgr.detect_faces(img_from, model=DetectedFace, align=align, consent=consent, commit=False))
    detected_faces_to   = list(rmgr.detect_faces(img_to,   model=DetectedFace, align=align, consent=consent, commit=False))
    if len(detected_faces_from) != 1:
        logging.error(f"Could not detect single face in provided image ({target_path_from}) !")
        sys.exit(1)
    if len(detected_faces_to) != 1:
        logging.error(f"Could not detect single face in provided image ({target_path_to}) !")
        sys.exit(1)
    _, detected_face_to   = detected_faces_to[0]
    _, detected_face_from = detected_faces_from[0]
    distance = ds.distance(detected_face_to.descriptor, detected_face_from.descriptor)
    if not threshold:
        if ds.threshold:
            threshold = ds.threshold
        else:
            threshold = ds.default_threshold
    if bool( distance <= threshold ):
        msg = "The 2 pictures contains the same face"
    else:
        msg = "The 2 pictures contains different faces"
    stdout(f"""{msg}
{len(msg) * "-"}
distance: {distance}
threshold: {threshold}
dataset: {ds}
""")


@cli.command()
@click.option("-h", "--host", default="127.0.0.1", help="Specify address to bind (default: 127.0.0.1)")
@click.option("-p", "--port", type=int, default=5000, help="Specify port to bind (default: 5000)")
@click.pass_context
def webui(ctx, host, port):
    app.run(host=host, port=port)


@cli.command()
@click.argument('face_id')
@click.pass_context
def face_delete(ctx, face_id):
    """
    Delete a face
    """
    face = db.session.query(Face).filter(Face.id==face_id).first()
    if not face:
        logging.error(f"face with id: {face_id} does not exist")
    else:
        db.session.delete(face)
        db.session.commit()

@cli.command()
@click.argument('image_id')
@click.pass_context
def image_delete(ctx, image_id):
    """
    Delete an image
    """
    image = db.session.query(Image).filter(Image.id==image_id).first()
    if not image:
        logging.error(f"image with id: {image_id} does not exist")
    else:
        db.session.delete(image)
        db.session.commit()

@cli.command()
@click.option("-o", "--output",     required=True,   default="backup.zip", help="Path to save archive (default: output.zip).")
@click.pass_context
def backup(ctx, output):
    """
    Backup full database and files. To restore use the following:

    \b
    ```bash
    unzip backup.zip -d /tmp/data
    export STORAGE_URI=file:///tmp/data
    export DATABASE_URI=sqlite:////tmp/data/recops.db
    recops dataset-list
    ``` 
    """
    with zipfile.ZipFile(output, mode="w") as archive:
        # backup database
        with open(database_uri.replace('sqlite:///', ''), 'rb') as f:
            archive.writestr('recops.db', f.read())
        # backup files
        for path in glob.glob(f"{storage_path}/*/*"):
            name = path.replace(storage_path, '')
            if name[0] == "/":
                name = name[1:]
            with open(path, 'rb') as f:
                archive.writestr(name, f.read())


@cli.command()
@click.option("-d", "--dataset-id", required=True,   help="Specify dataset id")
@click.option("-o", "--output",     required=True,   default="output.zip", help="Path to save archive (default: output.zip).")
@click.pass_context
def dataset_export(ctx, dataset_id, output):
    """
    Export datasets content into a zip archive
    """
    session = db.session
    with zipfile.ZipFile(output, mode="w") as archive:
        # Export faces with identities
        for identity in session.query(Identity)\
                            .filter(Identity.dataset_id==dataset_id)\
                            .all():
            for face in identity.faces:
                archive.writestr(
                    f"identities/{identity.name}/{face.checksum}.jpeg",
                    face.content)
        # Export faces without identities
        for face in session.query(Face)\
                            .filter(Face.dataset_id==dataset_id)\
                            .filter(~Face.identities.any())\
                            .all():
                archive.writestr(
                    f"faces/{face.checksum}.jpeg",
                    face.content)
        # Export images
        for image in session.query(Image)\
                            .filter(Image.dataset_id==dataset_id)\
                            .all():
                archive.writestr(
                    f"images/{image.checksum}.jpeg",
                    image.content)
        # Export detected-faces
        # for dface in session.query(DetectedFace)\
        #                     .filter(DetectedFace.dataset_id==dataset_id)\
        #                     .all():
        #         archive.writestr(
        #             f"detected-face/{dface.checksum}.jpeg",
        #             dface.content)


@cli.command()
@click.option("-d", "--dataset-id", required=True,   help="Specify dataset id")
@click.option("-o", "--output",     required=True,   default="clustered", help="Folder to export results.")
@click.option('--threshold',        default=None,    type=click.FLOAT,   help="Maximum distance to be used when comparing faces (unless specified the default threshold of basemodel is used)")
@click.option('--skip-single-faces', is_flag=True,   help="export only faces that match at least one face (default: False)")
@click.pass_context
def dataset_cluster_faces(ctx, dataset_id, output, threshold, skip_single_faces):
    """
    Cluster faces for given dataset
    """

    if not os.path.exists(output):
        os.makedirs(output)

    logging.debug(f"Loading dataset")
    ds = get_dataset_from_UUID(dataset_id)

    result = {
        "faces": {}, 
    }

    if not threshold:
        if ds.threshold:
            threshold = ds.threshold
        else:
            threshold = ds.default_threshold

    dataset_faces = db.session.query(Face)\
                        .filter(Face.dataset_id==ds.id)\
                        .options(load_only(Face.id, Face.descriptor))
    dataset_faces_length = len(list(dataset_faces))
    with click.progressbar( dataset_faces, length=dataset_faces_length, label=f" + Grouping faces ...") as faces:
        for face_1 in faces:

            logging.debug(f"Working on Face with ID:{face_1.id}")

            for face_2 in dataset_faces:
                if not face_1.id == face_2.id: # skip the same face during iteration
                    
                    # compute distance
                    distance = ds.distance(face_1.descriptor, face_2.descriptor)
                    
                    # Check if the 2 faces match
                    if bool( distance <= threshold ):

                        # Set idetity if already generated or generate a new one
                        if face_2.id in result["faces"]:
                            idn_id = result["faces"][face_2.id]
                        elif face_1.id in result["faces"]:
                            idn_id = result["faces"][face_1.id]
                        else:
                            idn_id = str(uuid.uuid4())

                        result["faces"][face_2.id] = idn_id
                        result["faces"][face_1.id] = idn_id
                        logging.debug(f"Face with ID:{face_1.id} matched face with ID:{face_2.id}")

            # If all faces iterated and no one matched then we set the identity as none
            if not face_1.id in result["faces"]:
                result["faces"][face_1.id] = None
                logging.debug(f"Face with ID {face_1.id} didnt match any other face")

    # Itetate for second time and actually export faces
    with click.progressbar( dataset_faces, length=dataset_faces_length, label=f" + Exporting faces ...") as faces:
        
        for face in faces:

            # load generated identity (or none) from previous iteration
            if face.id in result["faces"]:
                idn_id = result["faces"][face.id]
            
            # if no identity generated and we forced to export single faces,
            # then we generate an identity, otherwise we keep it empty. 
            if idn_id == None and ( not skip_single_faces ):
                idn_id = str(uuid.uuid4())

            # If we have an identity, then we export the face
            if idn_id:
                if not os.path.exists( os.path.join(output, idn_id) ):
                    os.makedirs(os.path.join(output, idn_id))
                with open( os.path.join(output, idn_id, f"{face.id}.jpeg" ), 'wb' ) as f:
                    f.write(face.content)



"""
@cli.command()
@click.pass_context
def db_schema(ctx):
    from sqlalchemy import MetaData
    from sqlalchemy_schemadisplay import create_schema_graph
    graph = create_schema_graph(metadata=MetaData(database_uri),
        show_datatypes=False, # The image would get nasty big if we'd show the datatypes
        show_indexes=False, # ditto for indexes
        rankdir='LR', # From left to right (instead of top to bottom)
        concentrate=False # Don't try to join the relation lines together
    )
    graph.write_png('/tmp/graph.png')

"""
