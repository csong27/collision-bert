import os

COLLISION_DIR = os.environ.get('COLLISION_DIR', '/hdd/song/collision/')
assert os.path.exists(COLLISION_DIR)

BOS_TOKEN = '[unused0]'
EOS_TOKEN = '[unused1]'

# Paraphrase related constants
PARA_DIR = os.path.join(COLLISION_DIR, 'paraphrase')

# BIRCH related constants
BIRCH_DIR = os.path.join(COLLISION_DIR, 'birch')
BIRCH_MODEL_DIR = os.path.join(BIRCH_DIR, 'models')
BIRCH_DATA_DIR = os.path.join(BIRCH_DIR, 'data')
BIRCH_INDEX_DIR = os.path.join(BIRCH_DIR, 'index')
BIRCH_PRED_DIR = os.path.join(BIRCH_DATA_DIR, 'predictions')
BIRCH_ALPHAS = [1.0, 0.5, 0.1]
BIRCH_GAMMA = 0.6

# Polyencoder related constants
PARLAI_DIR = os.path.join(COLLISION_DIR, 'parlai')
CONV2AI_PATH = os.path.join(PARLAI_DIR, 'personachat_self_original.json')

# PreSumm related constants
PRESUMM_DIR = os.path.join(COLLISION_DIR, 'presumm')
PRESUMM_DATA_DIR = os.path.join(PRESUMM_DIR, 'data')
PRESUMM_MODEL_DIR = os.path.join(PRESUMM_DIR, 'models')
PRESUMM_MODEL_PATH = os.path.join(PRESUMM_MODEL_DIR, 'bertext_cnndm_transformer_ckpt.pt')


# LM Model dir
BERT_LM_MODEL_DIR = os.path.join(COLLISION_DIR, 'wiki103', 'bert')
POLY_LM_MODEL_DIR = os.path.join(COLLISION_DIR, 'wiki103', 'polyencoder')
