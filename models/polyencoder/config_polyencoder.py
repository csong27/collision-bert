from parlai.core.opt import load_opt_file
from constant import PARLAI_DIR

PARLAI_MODEL_DIR = PARLAI_DIR + '/models/'
POLYENC_MODEL_DIR = PARLAI_MODEL_DIR + 'model_poly/'
POLYENC_OPT_FILE = POLYENC_MODEL_DIR + 'model.opt'
BI_MODEL_DIR = PARLAI_MODEL_DIR + 'model_bi/'
BIENC_OPT_FILE = BI_MODEL_DIR + 'model.opt'
PRETRAINED_BI_MODEL_DIR = PARLAI_MODEL_DIR + 'bi_model_huge_reddit/'
PRETRAINED_BIENC_OPT_FILE = PRETRAINED_BI_MODEL_DIR + 'model.opt'
PRETRAINED_POLY_MODEL_DIR = PARLAI_MODEL_DIR + 'poly_model_huge_reddit/'
PRETRAINED_POLYENC_OPT_FILE = PRETRAINED_POLY_MODEL_DIR + 'model.opt'


def load_poly_encoder_opt():
  opt = load_opt_file(POLYENC_OPT_FILE)
  if isinstance(opt['fixed_candidates_path'], str):
    opt['fixed_candidates_path'] = PARLAI_DIR + opt['fixed_candidates_path']
  opt['data_path'] = PARLAI_DIR + 'data'
  opt['datapath'] = PARLAI_DIR + 'data'
  opt['model_file'] = POLYENC_MODEL_DIR + 'model'
  opt['dict_file'] = POLYENC_MODEL_DIR + 'model.dict'
  opt['encode_candidate_vecs'] = False
  return opt


def load_bi_encoder_opt():
  opt = load_opt_file(BIENC_OPT_FILE)
  if isinstance(opt['fixed_candidates_path'], str):
    opt['fixed_candidates_path'] = PARLAI_DIR + opt['fixed_candidates_path']
  opt['data_path'] = PARLAI_DIR + 'data'
  opt['datapath'] = PARLAI_DIR + 'data'
  opt['model_file'] = BI_MODEL_DIR + 'model'
  opt['dict_file'] = BI_MODEL_DIR + 'model.dict'
  opt['encode_candidate_vecs'] = False
  return opt


def load_pretrained_poly_encoder_opt():
  opt = load_opt_file(PRETRAINED_POLYENC_OPT_FILE)
  if isinstance(opt['fixed_candidates_path'], str):
    opt['fixed_candidates_path'] = PARLAI_DIR + opt['fixed_candidates_path']
  opt['data_path'] = PARLAI_DIR + 'data'
  opt['datapath'] = PARLAI_DIR + 'data'
  opt['model_file'] = PRETRAINED_POLY_MODEL_DIR + 'model'
  opt['dict_file'] = PRETRAINED_POLY_MODEL_DIR + 'model.dict'
  opt['encode_candidate_vecs'] = False
  return opt


def load_pretrained_bi_encoder_opt():
  opt = load_opt_file(PRETRAINED_BIENC_OPT_FILE)
  if isinstance(opt['fixed_candidates_path'], str):
    opt['fixed_candidates_path'] = PARLAI_DIR + opt['fixed_candidates_path']
  opt['data_path'] = PARLAI_DIR + 'data'
  opt['datapath'] = PARLAI_DIR + 'data'
  opt['model_file'] = PRETRAINED_BI_MODEL_DIR + 'model'
  opt['dict_file'] = PRETRAINED_BI_MODEL_DIR + 'model.dict'
  opt['encode_candidate_vecs'] = False
  return opt
