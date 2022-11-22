from flask                     import Flask, url_for, send_from_directory, make_response
from flask_admin               import Admin, AdminIndexView, expose
from flask_admin.form          import thumbgen_filename
from flask_admin.contrib.sqla  import ModelView
from sqlalchemy                import func
# from jinja2                    import Markup
from markupsafe                import Markup
import io
import uuid

from recops.models            import *

os.makedirs(storage_path, exist_ok=True)
os.makedirs(f"{storage_path}/image", exist_ok=True)
os.makedirs(f"{storage_path}/face", exist_ok=True)

app = Flask(__name__,
            static_url_path='/static', 
            static_folder='static',)
app.config.update(
			SQLALCHEMY_TRACK_MODIFICATIONS=False,
			SQLALCHEMY_DATABASE_URI=database_uri)
app.config["SECRET_KEY"] = str(uuid.uuid4())
app.app_context().push()

# Initialize Sqlalchemy
db.init_app(app)
db.create_all()

# set optional bootswatch theme
app.config['FLASK_ADMIN_SWATCH'] = 'cerulean'


@app.route('/image/<image_id>/thumb', methods=('GET', 'HEAD'))
def image_thumb(image_id):
    """
    Route returning image thumb file
    """
    image = db.session.query(Image).filter(Image.id==image_id).first()
    if not image:
        abort(404)
    return send_from_directory(storage_path, thumbgen_filename(image.path), mimetype=image.content_type)

@app.route('/image/<image_id>/raw', methods=('GET', 'HEAD'))
def image_raw(image_id):
    """
    Route returning image thumb file
    """
    image = db.session.query(Image).filter(Image.id==image_id).first()
    if not image:
        abort(404)
    return send_from_directory(storage_path, image.path, mimetype=image.content_type)

@app.route('/detected_face/<face_id>/raw', methods=('GET', 'HEAD'))
def detected_face_raw(face_id):
    face = db.session.query(DetectedFace).filter(DetectedFace.id==face_id).first()
    face_img_array = PIL.Image.fromarray(face.img_array)
    face_img_array_raw = io.BytesIO()
    face_img_array.save(face_img_array_raw, format='JPEG')
    # return face_img_array_raw.getvalue()
    response = make_response(face_img_array_raw.getvalue())
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set('Content-Disposition', 'attachment', filename=f'{str(face.id)}.jpg')
    return response

@app.route('/face/<face_id>/raw', methods=('GET', 'HEAD'))
def face_raw(face_id):
    face = db.session.query(Face).filter(Face.id==face_id).first()
    
    face_img_array = PIL.Image.fromarray(face.img_array)
    face_img_array_raw = io.BytesIO()
    face_img_array.save(face_img_array_raw, format='JPEG')
    # return face_img_array_raw.getvalue()
    response = make_response(face_img_array_raw.getvalue())
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set('Content-Disposition', 'attachment', filename=f'{str(face.id)}.jpg')
    return response


class RecopsIndexView(AdminIndexView):
    def is_visible(self):
        # This view won't appear in the menu structure
        return False
    # @expose('/')
    # def index(self):
    #     # return self.render('admin/dashboard.html')
    #     return self.render(self._template)

admin = Admin(
    app,
    name="recops",
    # url="/",
    template_mode='bootstrap3',
    index_view=RecopsIndexView(url="/")
)

class IdentityView(ModelView):
    page_size = 50
    column_list            = ('id', 'name', 'description', 'dataset_id')
    column_searchable_list = ('id', 'name', 'description', 'dataset_id')
    column_filters         = ['dataset', 'faces']


class DetectedFaceView(ModelView):
    can_create = False
    page_size = 50
    def _list_thumbnail(view, context, model, name):
        return Markup(f'<img src="{url_for("detected_face_raw", face_id=model.id)}" style="max-width:100%;height:auto;" >')
    column_list            = ('id', 'img_array', 'gender', 'image_id', 'dataset_id', 'checksum')
    column_searchable_list = ('id', 'image_id', 'checksum')
    column_filters         = ['dataset', 'gender', 'checksum']
    column_formatters = {
        'img_array': _list_thumbnail
    }

class FaceView(ModelView):
    can_create = False
    page_size = 50

    def _list_face_img(view, context, model, name):
        # if not model.path:
        #     return ''
        # return Markup(f'''
        # <a href="{url_for("face_svg", face_id=model.id)}">
        #     <img src="{url_for("raw_path", model=model.__tablename__, model_id=model.id)}" style="max-width:25%;height:auto;">
        # </a>
        # ''')
        return Markup(f'<img src="{url_for("face_raw", face_id=model.id)}" style="max-width:100%;height:auto;" >')

    def _list_dataset_id(view, context, model, name):
        return Markup(f'''
            <a href="{ url_for('dataset.index_view') }">{ model.dataset.name }</a>
        ''')

    def _list_identities(view, context, model, name):
        if model.identities:
            return Markup("".join([
                f"<a href=\"{ url_for('identity.index_view') }?search={ idn.name }\">{ idn.name }</a><br>" for idn in model.identities
            ]))
        return ""

    column_list            = ('id', 'img_array', 'gender', 'emotion', 'age', 'race', 'image.checksum', 'dataset', 'identities')
    column_searchable_list = ('id', 'image_id', 'checksum',)
    column_filters         = ['dataset', 'identities']
    column_formatters = {
        'img_array': _list_face_img,
        'dataset': _list_dataset_id,
        'identities': _list_identities,
    }

class ImageView(ModelView):
    page_size = 50
    can_create = False
    def _list_thumbnail(view, context, model, name):
        return Markup(f'''
            <a href="{ url_for("image_raw", image_id=model.id) }">
                <img src="{url_for("image_thumb", image_id=model.id)}" style="max-width:100%;height:auto;" >
            </a>
        ''')
    column_list            = ('id', 'img_array', 'path', 'dataset_id', 'checksum')
    column_searchable_list = ('id', 'path', 'checksum')
    column_filters         = [ "dataset", "faces", "detected_faces" ]
    column_formatters = {
        'img_array': _list_thumbnail
    }
    def get_query(self):
        return self.session.query(self.model).filter(self.model.marked=='IMPORT')
    def get_count_query(self):
        return self.session.query(func.count('*')).filter(self.model.marked=='IMPORT')

class DetectedImageView(ImageView):
    def get_query(self):
        return self.session.query(self.model).filter(self.model.marked=='DETECT')
    def get_count_query(self):
        return self.session.query(func.count('*')).filter(self.model.marked=='DETECT')

class DatasetView(ModelView):
    page_size = 50
    form_excluded_columns = ['faces', 'detected_faces', 'images','jobs', 'identities']
    form_choices = {
        'detector_backend': [
            ( "opencv",     "OpenCV"),
            ( "ssd",        "Ssd"),
            ( "dlib",       "Dlib"),
            ( "mtcnn",      "Mtcnn"),
            ( "retinaface", "RetinaFace"),
            ( "mediapipe",  "Mediapipe" ),
        ],
        'basemodel_backend': [
            ( "ArcFace",     'ArcFace'),
            ( "Boosting",    'Boosting'),
            ( "DeepID",      'DeepID'),
            ( "DlibResNet",  'Dlib ResNet'),
            ( "DlibWrapper", 'Dlib'),
            ( "Facenet",     'Facenet'),
            ( "Facenet512",  'Facenet512'),
            ( "FbDeepFace",  'FbDeepFace'),
            ( "OpenFace",    'OpenFace'),
            ( "VGGFace",     'VGGFace'),
        ]
    }
    # column_editable_list = [ 'name', 'threshold', 'distance_metric' ]
    column_list = ('id', 
                    'name',
                    'detector_backend',
                    'basemodel_backend',
                    'threshold',
                    'distance_metric',)


class MatchedFaceAssociationView(ModelView):
    page_size = 50

    def _list_detected_thumbnail(view, context, model, name):
        return Markup(f'''
            <img src="{url_for("detected_face_raw", face_id=model.detected_face_id)}"  style="max-width:100%;height:auto;" >
        ''')

    def _list_thumbnail(view, context, model, name):
        return Markup(f'<img src="{url_for("face_raw", face_id=model.face_id)}" style="max-width:100%;height:auto;" >')

    column_list       = ('detected_face_id', 'detected_face_img_array', 'face_id', 'face_img_array', 'distance', 'threshold')
    column_formatters = {
        'detected_face_img_array': _list_detected_thumbnail,
        'face_img_array': _list_thumbnail
    }
    column_filters    = ['face', 'face.identities', 'detected_face']


# Add administrative views here
admin.add_view(DatasetView(Dataset, db.session))
admin.add_view(DetectedFaceView(DetectedFace, db.session, name="Detected Face", endpoint="detected_face"))
admin.add_view(DetectedImageView(Image, db.session, name="Detect Image", endpoint="detect_image"))
admin.add_view(FaceView(Face, db.session))
admin.add_view(ImageView(Image, db.session, name="Face Image", endpoint="face_image" ))
admin.add_view(IdentityView(Identity, db.session, endpoint="identity" ))
admin.add_view(MatchedFaceAssociationView(MatchedFaceAssociation, db.session))
# admin.add_view(ModelView(Job, db.session))
