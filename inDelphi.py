from __future__ import division
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle, copy
from scipy.stats import entropy

model = None
nn_params = None
nn2_params = None
normalizer = None
rate_model = None
bp_model = None


##
# NN
##
def sigmoid(x):
  return 0.5 * (np.tanh(x) + 1.0)

def nn_function(params, inputs):
  # """Params is a list of (weights, bias) tuples.
  #    inputs is an (N x D) matrix."""
  inpW, inpb = params[0]
  inputs = sigmoid(np.dot(inputs, inpW) + inpb)
  for W, b in params[1:-1]:
    outputs = np.dot(inputs, W) + b
    inputs = sigmoid(outputs)
  outW, outb = params[-1]
  outputs = np.dot(inputs, outW) + outb
  return outputs.flatten()

##
# Sequence featurization
##
def get_gc_frac(seq):
  return (seq.count('C') + seq.count('G')) / len(seq)

def find_microhomologies(left, right):
  start_idx = max(len(right) - len(left), 0)
  mhs = []
  mh = [start_idx]
  for idx in range(min(len(right), len(left))):
    if left[idx] == right[start_idx + idx]:
      mh.append(start_idx + idx + 1)
    else:
      mhs.append(mh)
      mh = [start_idx + idx +1]
  mhs.append(mh)
  return mhs

def featurize(seq, cutsite, DELLEN_LIMIT = 60):
  # print 'Using DELLEN_LIMIT = %s' % (DELLEN_LIMIT)
  mh_lens, gc_fracs, gt_poss, del_lens = [], [], [], []
  for del_len in range(1, DELLEN_LIMIT):
    left = seq[cutsite - del_len : cutsite]
    right = seq[cutsite : cutsite + del_len]

    mhs = find_microhomologies(left, right)
    for mh in mhs:
      mh_len = len(mh) - 1
      if mh_len > 0:
        gtpos = max(mh)
        gt_poss.append(gtpos)

        s = cutsite - del_len + gtpos - mh_len
        e = s + mh_len
        mh_seq = seq[s : e]
        gc_frac = get_gc_frac(mh_seq)

        mh_lens.append(mh_len)
        gc_fracs.append(gc_frac)
        del_lens.append(del_len)

  return mh_lens, gc_fracs, gt_poss, del_lens

##
# Prediction
##
def predict(seq, cutsite):
  # Predict 1 bp insertions and all deletions (MH and MH-less)
  # Most complete "version" of inDelphi
  # Requires rate_model (k-NN) to predict 1 bp insertion rate compared to deletion rate
  # Also requires bp_model to predict 1 bp insertion genotype given -4 nucleotide

  ################################################################
  #####
  ##### Predict MH and MH-less deletions
  #####
  # Predict MH deletions

  if nn_params == None:
    init_model()

  mh_len, gc_frac, gt_pos, del_len = featurize(seq, cutsite)

  # Form inputs
  pred_input = np.array([mh_len, gc_frac]).T
  del_lens = np.array(del_len).T
  
  # Predict
  mh_scores = nn_function(nn_params, pred_input)
  mh_scores = mh_scores.reshape(mh_scores.shape[0], 1)
  Js = del_lens.reshape(del_lens.shape[0], 1)
  unfq = np.exp(mh_scores - 0.25*Js)

  # Add MH-less contribution at full MH deletion lengths
  mh_vector = np.array(mh_len)
  mhfull_contribution = np.zeros(mh_vector.shape)
  for jdx in range(len(mh_vector)):
    if del_lens[jdx] == mh_vector[jdx]:
      dl = del_lens[jdx]
      mhless_score = nn_function(nn2_params, np.array(dl))
      mhless_score = np.exp(mhless_score - 0.25*dl)
      mask = np.concatenate([np.zeros(jdx,), np.ones(1,) * mhless_score, np.zeros(len(mh_vector) - jdx - 1,)])
      mhfull_contribution = mhfull_contribution + mask
  mhfull_contribution = mhfull_contribution.reshape(-1, 1)
  unfq = unfq + mhfull_contribution

  # Store predictions to combine with mh-less deletion preds
  pred_del_len = copy.copy(del_len)
  pred_gt_pos = copy.copy(gt_pos)

  ################################################################
  #####
  ##### Predict MH and MH-less deletions
  #####
  # Predict MH-less deletions
  mh_len, gc_frac, gt_pos, del_len = featurize(seq, cutsite)

  unfq = list(unfq)

  pred_mhless_d = defaultdict(list)
  # Include MH-less contributions at non-full MH deletion lengths
  nonfull_dls = []
  for dl in range(1, 60):
    if dl not in del_len:
      nonfull_dls.append(dl)
    elif del_len.count(dl) == 1:
      idx = del_len.index(dl)
      if mh_len[idx] != dl:
        nonfull_dls.append(dl)
    else:
        nonfull_dls.append(dl)

  mh_vector = np.array(mh_len)
  for dl in nonfull_dls:
    mhless_score = nn_function(nn2_params, np.array(dl))
    mhless_score = np.exp(mhless_score - 0.25*dl)

    unfq.append(mhless_score)
    pred_gt_pos.append('e')
    pred_del_len.append(dl)

  unfq = np.array(unfq)
  total_phi_score = float(sum(unfq))

  nfq = np.divide(unfq, np.sum(unfq))  
  pred_freq = list(nfq.flatten())

  d = {'Length': pred_del_len, 'Genotype Position': pred_gt_pos, 'Predicted_Frequency': pred_freq}
  pred_del_df = pd.DataFrame(d)
  pred_del_df['Category'] = 'del'

  ################################################################
  #####
  ##### Predict Insertions
  #####
  # Predict 1 bp insertions
  dlpred = []
  for dl in range(1, 28+1):
    crit = (pred_del_df['Length'] == dl)
    dlpred.append(sum(pred_del_df[crit]['Predicted_Frequency']))
  dlpred = np.array(dlpred) / sum(dlpred)
  norm_entropy = entropy(dlpred) / np.log(len(dlpred))
  precision = 1 - norm_entropy

  fiveohmapper = {'A': [1, 0, 0, 0], 
                  'C': [0, 1, 0, 0], 
                  'G': [0, 0, 1, 0], 
                  'T': [0, 0, 0, 1]}
  threeohmapper = {'A': [1, 0, 0, 0], 
                   'C': [0, 1, 0, 0], 
                   'G': [0, 0, 1, 0], 
                   'T': [0, 0, 0, 1]}
  fivebase = seq[cutsite - 1]
  threebase = seq[cutsite]
  onebp_features = fiveohmapper[fivebase] + threeohmapper[threebase] + [precision] + [total_phi_score]
  for idx in range(len(onebp_features)):
    val = onebp_features[idx]
    onebp_features[idx] = (val - normalizer[idx][0]) / normalizer[idx][1]
  onebp_features = np.array(onebp_features).reshape(1, -1)
  rate_1bpins = float(rate_model.predict(onebp_features))

  # Predict 1 bp genotype frequencies
  pred_1bpins_d = defaultdict(list)
  negfivebase = seq[cutsite - 2]
  negfourbase = seq[cutsite - 1]
  negthreebase = seq[cutsite]

  for ins_base in bp_model[negfivebase][negfourbase][negthreebase]:
    freq = bp_model[negfivebase][negfourbase][negthreebase][ins_base]
    freq *= rate_1bpins / (1 - rate_1bpins)
    pred_1bpins_d['Category'].append('ins')
    pred_1bpins_d['Length'].append(1)
    pred_1bpins_d['Inserted Bases'].append(ins_base)
    pred_1bpins_d['Predicted_Frequency'].append(freq)

  pred_1bpins_df = pd.DataFrame(pred_1bpins_d)
  pred_all_df = pred_del_df.append(pred_1bpins_df, ignore_index = True)
  pred_all_df['Predicted_Frequency'] /= sum(pred_all_df['Predicted_Frequency'])

  return pred_del_df, pred_all_df, total_phi_score, rate_1bpins


##
# Init
##
def init_model(run_iter = 'abf', param_iter = 'aag'):
  if nn_params != None:
    return

  print 'Initializing model %s/%s...' % (run_iter, param_iter)

  global nn_params
  global nn2_params
  nn_params = pickle.load(open('%s_%s_nn.pkl' % (run_iter, 
                                              param_iter,
                                              )))
  nn2_params = pickle.load(open('%s_%s_nn2.pkl' % (run_iter, 
                                              param_iter,
                                              )))

  global normalizer
  global rate_model
  global bp_model
  bp_model = pickle.load(open('bp_model_%s.pkl' % ('v3')))
  rate_model = pickle.load(open('rate_model_%s.pkl' % ('v3')))
  normalizer = pickle.load(open('normalizer_%s.pkl' % ('v3')))

  print 'Done'
  return

