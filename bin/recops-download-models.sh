#!/bin/bash
# recops-download-models.sh

OUTPUT=~/.deepface/weights

mkdir -p $OUTPUT

curl -L -o $OUTPUT/age_model_weights.h5 \
		https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5

curl -L -o $OUTPUT/arcface_weights.h5 \
		https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5

curl -L -o $OUTPUT/deepid_keras_weights.h5 \
		https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_keras_weights.h5

curl -L -o $OUTPUT/facenet512_weights.h5 \
		https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5

curl -L -o $OUTPUT/facenet_weights.h5 \
		https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5

curl -L -o $OUTPUT/facial_expression_model_weights.h5 \
		https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5

curl -L -o $OUTPUT/gender_model_weights.h5 \
		https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5

curl -L -o $OUTPUT/openface_weights.h5 \
		https://github.com/serengil/deepface_models/releases/download/v1.0/openface_weights.h5

curl -L -o $OUTPUT/race_model_single_batch.h5 \
		https://github.com/serengil/deepface_models/releases/download/v1.0/race_model_single_batch.h5

curl -L -o $OUTPUT/retinaface.h5 \
		https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5

curl -L -o $OUTPUT/vgg_face_weights.h5 \
		https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5 

curl -L -o $OUTPUT/face_recognition_sface_2021dec.onnx \
		https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

curl -L -o $OUTPUT/age_model_weights.h5 \
	https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5

curl -L -o $OUTPUT/gender_model_weights.h5 \
	https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5

curl -L -o $OUTPUT/facial_expression_model_weights.h5 \
	https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5

curl -L -o $OUTPUT/race_model_single_batch.h5 \
	https://github.com/serengil/deepface_models/releases/download/v1.0/race_model_single_batch.h5
