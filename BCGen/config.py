import texar as tx
#784
grudim = 768
bertdim = 768
dcoder_config = {
    'num_blocks': 12,
    'dim': bertdim,
    'multihead_attention': {
        'num_heads': 8,
        'output_dim': bertdim
        # See documentation for more optional hyperparameters
    },
    'position_embedder_hparams': {
        'dim': bertdim
    },
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
        },
    },
    'poswise_feedforward': tx.modules.default_transformer_poswise_net_hparams(
        output_dim=bertdim)
}

loss_label_confidence = 0.9

random_seed = 1234
beam_width = 5
alpha = 0.6
hidden_dim = bertdim


opt = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8
        }
    }
}

#warmup steps must be 0.1% of number of iterations
lr = {
    'learning_rate_schedule': 'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
    'lr_constant': 2 * (hidden_dim ** -0.5),
    'static_lr': 1e-3,
    'warmup_steps': 10000,
}

bos_token_id = 101
eos_token_id = 102

model_dir = "./models_10w_fuse_emb"
run_mode = "train_and_evaluate"

batch_size = 32
eval_batch_size = 32
test_batch_size = 32
# batch_size = 16
# eval_batch_size = 16
# test_batch_size = 16

max_train_steps = 100000

display_steps = 10
checkpoint_steps = 2000
eval_steps = 10000
test_steps = 10000

max_decoding_length = 400

max_seq_length_src = 384 # 150
max_seq_length_cfg = 384 # 150
max_seq_length_tgt = 30 # 30

epochs =10

is_distributed = False

data_dir = "10w_cfgsbt_code2/"

train_out_file = "10w_cfgsbt_code2/train.tf_record"
eval_out_file = "10w_cfgsbt_code2/eval.tf_record"
test_out_file = "10w_cfgsbt_code2/test.tf_record"

bert_pretrain_dir="./bert_uncased_model"

train_story = "10w_cfgsbt_code2/train_story.txt"
train_summ = "10w_cfgsbt_code2/train_summ.txt"
train_cfg = "10w_cfgsbt_code2/train_cfg.txt"

eval_story = "10w_cfgsbt_code2/eval_story.txt"
eval_summ = "10w_cfgsbt_code2/eval_summ.txt"
eval_cfg = "10w_cfgsbt_code2/eval_cfg.txt"

test_story = "10w_cfgsbt_code2/test_story.txt"
test_cfg = "10w_cfgsbt_code2/test_cfg.txt"
test_summ = "10w_cfgsbt_code2/test_summ.txt"


