import os 
import numpy as np 
import flask
import tarfile, boto3
import json

from flask import Flask, request
from recsys.ap_recsys import ApRecsys
from recsys.train.aurora_client import AuroraConfig
from recsys.serve.dynamodb_client import DynamoDBClient
from logger import logger

lg = logger().logger

def download_artifacts(train_job_name, model_save_path):
    # s3 configure
    s3 = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bucket = s3.Bucket(BUCKET_NAME)

    # make directory that model will save
    if (os.path.isdir(model_save_path) == False):
        os.mkdir(model_save_path)

    s3_Path = os.path.join(ARTIFACT_PATH, train_job_name.upper())

    out_Path = model_save_path + 'model.tar.gz'

    for file in bucket.objects.filter(Prefix=s3_Path):
        s3_client.download_file(file.bucket_name, file.key, out_Path)
        lg.debug("Download_file ... ", file.key)
    compressed = tarfile.open(out_Path)
    compressed.extractall(path=model_save_path)

def get_api_server(ap_model, redis_client, top_k):
    app = Flask(__name__)
 
    @app.route('/ping', methods=['GET'])
    def ping():
        """ Sagemaker Health Check...loading is okay?"""
        return flask.Response(response='\n', status=200, mimetype='application/json')

    @app.route('/invocations', methods=['POST'])
    def get_personal_recommendation():

        content = request.json
        userId = content['userId']

        key = f'userId:{userId}'

        input_itemId_seq = redis_client[key]
        input_itemId_seq.reverse()
        input_itemId_seq = [itemId.decode("utf-8") for itemId in input_itemId_seq]

        input_index_seq = [model.get_index(itemId) for itemId in input_itemId_seq]
        while (-1 in input_index_seq): del input_index_seq[input_index_seq.index(-1)]
        input_itemId_seq = [model.get_itemId(itemIndex) for itemIndex in input_index_seq]


        if len(input_itemId_seq) == 0:
            response = {
                'message': 'user history does not exist'
            }
            result = json.dumps(response)
            return flask.Response(response=result, status=200, mimetype='applicaton/json')

        input_index_seq = [ap_model.get_index(itemId) for itemId in input_itemId_seq]

        pad_input_items = np.zeros(ap_model.max_seq_len, np.int32)
        pad_input_items[:len(input_index_seq)] = input_index_seq
        input = np.zeros(1, dtype=[('seq_item_id', (np.int32, ap_model.max_seq_len)), ('seq_len', np.int32)])
        input[0] = (pad_input_items, len(input_index_seq))
        user_embedding, logits = ap_model.serve(input)

        user_embedding = np.squeeze(user_embedding)
        logit = np.squeeze(logits)

        item_index = np.argsort(logit)[::-1][:top_k]

        recommendation_itemIds = []
        for index in item_index:
            recommendation_itemIds.append(ap_model.get_itemId(index))

        items_info = ap_model.get_item_info(input_itemId_seq)
        recommendation_items_info = ap_model.get_item_info(recommendation_itemIds)

        response = {
            'userId': userId,
            'history_items_info': items_info,
            'recommendation_items_info': recommendation_items_info
        }
        dump_response = json.dumps(response)

        return flask.Response(response=dump_response, status=200, mimetype='applicaton/json')

    return app

def serve():
    aurora_config 
    
    # DynamoDB 객체 생성
    dynamoDB_client = DynamoDBClient()

    train_job_name = os.environ['TRAIN_JOB_NAME']
    model_save_path = './model_artifacts/'
    download_artifacts(train_job_name, model_save_path)

    ap_model = ApRecsys(model_save_path,  aurora_config)
    ap_model.build_serve_model()
    ap_model.restore(restore_serve=True) 

    # WAS
    api_server = get_api_server(ap_model, dynamoDB_client, top_k=config().json_data["System"]["top_k"])
    return api_server
    
app = serve()