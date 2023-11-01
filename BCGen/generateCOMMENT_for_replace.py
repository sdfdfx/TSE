#!/usr/bin/env python
# @Project ：test 
# @File    ：generateCOMMENT2.py
# @Author  ：
# @Date    ：2022/4/18 16:49 
# 
# --------------------------------------------------------
import sys

if not 'texar_repo' in sys.path:
    sys.path += ['texar_repo']
import tensorflow as tf
import texar as tx
import numpy as np
from config import *
from replace_model import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if not os.path.exists('./output'):
    os.mkdir('./output')
def _eval_epoch(sess, epoch, mode):
    cfg, diff, references, hypotheses = [], [], [], []

    if mode == 'test':
        iterator.restart_dataset(sess, 'test')
        bsize = test_batch_size
        fetches = {
            'inferred_ids': inferred_ids,
        }
        bno = 0

    with open('./output/refs_10w_cfg3.txt', 'w') as f1, open('./output/hyps_10w_cfg3.txt', 'w') as f2,\
            open('./output/srcs_10w_cfg3.txt', 'w') as f3:
        while True:

            # print("Temp",temp)
            try:
                print("Batch", bno)
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'test'),
                    tx.global_mode(): tf.estimator.ModeKeys.PREDICT,
                }
                op = sess.run([batch], feed_dict)

                feed_dict = {
                    src_input_ids: op[0]['src_input_ids'],
                    src_segment_ids: op[0]['src_segment_ids'],
                    tx.global_mode(): tf.estimator.ModeKeys.PREDICT
                }
                fetches_ = sess.run(fetches, feed_dict=feed_dict)
                labels = op[0]['tgt_labels']
                diffs = op[0]['src_input_ids']
                hypotheses.extend(h.tolist() for h in fetches_['inferred_ids'])
                references.extend(r.tolist() for r in labels)
                diff.extend(d.tolist() for d in diffs)

                bno = bno + 1

            except tf.errors.OutOfRangeError:
                break

        hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
        references = utils.list_strip_eos(references, eos_token_id)
        diff = utils.list_strip_eos(diff, eos_token_id)

        for example in range(len(references)):
            hwords = tokenizer.convert_ids_to_tokens(hypotheses[example])
            rwords = tokenizer.convert_ids_to_tokens(references[example])
            dwords = tokenizer.convert_ids_to_tokens(diff[example])

            hwords = tx.utils.str_join(hwords).replace(" ##", "")
            rwords = tx.utils.str_join(rwords).replace(" ##", "")
            dwords = tx.utils.str_join(dwords).replace(" ##", "").replace("[CLS]", "")

            f1.write(rwords + '\n')
            f2.write(hwords + '\n')
            f3.write(dwords + '\n')


tx.utils.maybe_create_dir(model_dir)
logging_file = os.path.join(model_dir, "logging.txt")
logger = utils.get_logger(logging_file)

# with tf.device('/gpu:0'):
with tf.Session() as sess:  # config=config
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())

    smry_writer = tf.summary.FileWriter(model_dir, graph=sess.graph)

    if run_mode == 'train_and_evaluate':
        logger.info('Begin running with train_and_evaluate mode')

        if tf.train.latest_checkpoint(model_dir) is not None:
            print("*"*100, tf.train.latest_checkpoint(model_dir))
            logger.info('Restore latest checkpoint in %s' % model_dir)
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        iterator.initialize_dataset(sess)

        # iterator.restart_dataset(sess, 'test')
        step = _eval_epoch(sess, 0, 'test')

    else:
        raise ValueError('Unknown mode: {}'.format(run_mode))