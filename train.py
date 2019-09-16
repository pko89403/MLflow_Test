# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
from termcolor import colored
from datetime import datetime as dt
import datetime
import csv
import time
import sys
import mlflow as mf
import json

from recsys.ap_recsys import ApRecsys
from recsys.evaluators.auc import AUC
from recsys.evaluators.precision import Precision
from recsys.evaluators.recall import Recall
from recsys.train.aurora_client import AuroraConfig, AuroraClient
from recsys.serve.dynamodb_client import DynamoDBClient, DynamoDBConnectionConfig
from collections import OrderedDict
from logger import logger
from config import config

lg = logger().logger
lg.debug('tensorflow version: ' + str(tf.__version__))
lg.debug('numpy version: ' + str(np.__version__))
lg.debug('mlflow version: ' + mf.__version__)



# should be run by Mlflow
def train():

    with mf.start_run() as run:

        start_time = time.time()
        ap_recsys = ApRecsys(model_save_path, db_config)

        # Hyper-param when init ap_recsys
        mf.log_param("item_embedding_dimension", config().json_data["Hyper_Parameter"]["item_embedding_dimension"])
        mf.log_param("max_seq_len", config().json_data["Hyper_Parameter"]["max_seq_len"])
        mf.log_param("batch_size", config().json_data["Hyper_Parameter"]["batch_size"])
        mf.log_param("eval_iter", config().json_data["Hyper_Parameter"]["eval_iter"])
        mf.log_param("eval_percentage", config().json_data["Hyper_Parameter"]["eval_percentage"])
        mf.log_param("learning_rate", config().json_data["Hyper_Parameter"]["learning_rate"])

        train_sampler = ap_recsys.get_train_sampler()
        eval_sampler = ap_recsys.get_eval_sampler()

        train_model = ap_recsys.build_train_model()
        serve_model = ap_recsys.build_serve_model()

        # get auroraDB data
        mf.log_param('auroraDB_total_items', ap_recsys.aurora_total_items)
        mf.log_param('auroraDB_total_users', ap_recsys.aurora_total_users)

        ap_recsys.add_evaluator(Precision(precision_at=[100]))
        ap_recsys.add_evaluator(Recall(recall_at=[50, 100, 150, 200, 250]))
        ap_recsys.add_evaluator(AUC())

        acc_loss = 0
        min_loss = None
        total_iter = 0

        while True:
            summary = tf.Summary()
            batch_data = train_sampler.next_batch()
            loss = ap_recsys.train(total_iter, batch_data)

            if min_loss is None:
                min_loss = loss
                acc_loss = loss

            if loss < min_loss:
                min_loss = loss

            acc_loss += loss
            total_iter += 1

            summary.value.add(tag='min_loss', simple_value=min_loss)

            mf.log_metric('total_iter', total_iter)
            mf.log_metric('min_loss', min_loss)

            metric_avg_loss_tag = 'Mean_loss per ' + str(ap_recsys.eval_iter)
            metric_auc_tag = 'AUC per ' + str(ap_recsys.eval_iter)
            metric_rank_above_tag = 'Rank_above per ' + str(ap_recsys.eval_iter)

            if total_iter % ap_recsys.eval_iter == 0:
                avg_loss = acc_loss / ap_recsys.eval_iter
                lg.debug(str(total_iter) + "avg_loss:" + str(avg_loss))
                lg.debug("total_iter" + total_iter.__str__() + "avg_loss:" + avg_loss.__str__())

                summary.value.add(tag='avg_loss', simple_value=avg_loss)
                acc_loss = 0

                eval_results = ap_recsys.evaluate(eval_sampler=eval_sampler, step=total_iter)
                eval_results = dict(eval_results)

                result_stdout = ''
                for result in eval_results:
                    average_result = np.mean(eval_results[result], axis=0)
                    result_stdout += f'[{result}] {average_result} '

                lg.debug(result_stdout)

                # MLFLOW LOG_METRIC
                mf.log_metric(metric_avg_loss_tag, avg_loss)
                mf.log_metric(metric_auc_tag, np.mean(eval_results['AUC']))
                mf.log_metric(metric_rank_above_tag, np.mean(eval_results['rank_above']))

                # MLFLOW LOG_ARTIFACTS ( DIrectory)
                mf.log_artifacts(model_save_path)  # save all files at model_save_path

                ap_recsys.train_writer.add_summary(summary, total_iter)
                elaped_time = int(time.strftime("%H", time.gmtime(time.time() - start_time)))

                if elaped_time > 3:
                    break

    mf.end_run()
    print("elaped time", time.strftime("%H:%M:%S", time.gmtime(elaped_time)))

if __name__ == '__main__':
    train()
