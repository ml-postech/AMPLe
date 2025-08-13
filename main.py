


import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import shutil
import torch as th
import wandb

# from accelerate import Accelerator
from accelerate.utils import set_seed
from copy import deepcopy
from datasets import Dataset, load_dataset, concatenate_datasets
from torch import nn
from torch.nn.functional import logsigmoid, relu
from torch.distributions import Multinomial, Categorical
from torch.distributions.dirichlet import Dirichlet
device = "cpu"


SMALL_SIZE = 10
MEDIUM_SIZE = 11
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


def compute_probs(w, a, b, beta=None, return_diff=False, log=False):
  """"""

  deterministic = (beta is None)
  beta = beta or 1.0
  diff = beta * w @ (a - b).T
  dummy = th.zeros_like(diff)
  kwargs = dict(dtype=th.float, device=device)
  if not deterministic:
    if log:
      probs = (
        th.stack([logsigmoid(-diff),
                  th.zeros_like(diff),
                  logsigmoid(+diff)], axis=-1).to(**kwargs))
    else:
      probs = (
        th.stack([1 / (1 + th.exp(+diff)),
                  dummy,
                  1 / (1 + th.exp(-diff))], axis=-1).to(**kwargs))
  else:
    if log:
      raise ValueError("cannot compute log-likelihood")
    tie = th.abs(diff) <= 0.0
    probs = (
      th.stack([((~tie) & (diff < 0)) + (tie * 0.5),
                 dummy,
                 ((~tie) & (diff > 0)) + (tie * 0.5)], axis=-1).to(**kwargs))
  return ((probs, diff) if return_diff else probs)


def generate_feedback(w, a, b, beta=None):
  """"""
  probs, diff = compute_probs(w, a, b, beta=beta, return_diff=True)
  if len(probs.shape) == 2:
    return th.multinomial(probs, 1)[:, 0] - 1, diff
  cat = Categorical(probs)
  return cat.sample() - 1, diff


def compute_likelihood(w, a, b, y, *, beta, log=False):
  """"""
  probs = compute_probs(w, a, b, beta=beta, log=log)
  if len(probs.shape) == 2:
    return probs[th.arange(len(probs)), y + 1]

  W, N, _ = probs.shape
  i1 = th.arange(W)[..., np.newaxis]
  i2 = th.arange(N)[np.newaxis, ...]
  return probs[i1, i2, y + 1]


def metropolis_hastings(target,
                        proposal,
                        init,
                        n_samples,
                        burnin,
                        thin):
  """"""

  samples = []
  curr = init
  for i in range(burnin + thin * n_samples - 1):
    next = proposal.sample(given=curr)
    metropolis_ratio = target.prob(next) / target.prob(curr)
    hastings_ratio = (proposal.prob(curr, given=next) / proposal.prob(next, given=curr))
    if (np.random.uniform(0, 1) < 
        min(1, metropolis_ratio * hastings_ratio)):
      curr = next
    samples.append(curr)
  return th.stack(samples[burnin::thin], axis=0).to(device=device)


class Proposal:

  def __init__(self, jitter, concentration, propose):
    self.jitter = jitter
    self.step = concentration
    self.propose = propose

  def sample(self, given):
    given = given + self.jitter
    given = given / given.sum(axis=-1, keepdim=True)
    given = given * self.step
    if self.propose == "skewd":
      dirichlet = Dirichlet(given)
    else:
      dirichlet = Dirichlet(th.ones_like(given))#(given)
    return dirichlet.sample()

  def prob(self, x, given):
    given = given + self.jitter
    given = given / given.sum(axis=-1, keepdim=True)
    given = given * self.step
    if self.propose == "skewd":
      dirichlet = Dirichlet(given)
    elif self.propose == "uniform":
      dirichlet = Dirichlet(th.ones_like(given))#(given)
    else:
      raise
    return dirichlet.log_prob(x).exp()


class Target:

  @classmethod
  def from_data(cls,
                prior,
                beta,
                eps,
                gamma,
                items_a,
                items_b,
                comparisons,
                t):
    self = cls(prior, beta, eps, gamma)
    self.items_a = items_a[:t]
    self.items_b = items_b[:t]
    self.comparisons = comparisons[:t]
    return self

  def __init__(self, prior, beta, eps, gamma):

    self.prior = prior
    self.beta = beta
    self.eps = eps
    self.gamma = gamma

    self.items_a = []
    self.items_b = []
    self.comparisons = []

  def prob(self, x):
    prior = self.prior(x)
    if len(self.comparisons) <= 0:
      return prior
    return prior * self.likelihood(x)

  def likelihood(self, preference):
    """"""
    likelihood = (
      compute_likelihood(preference,
                         th.stack(self.items_a, axis=0).to(device),
                         th.stack(self.items_b, axis=0).to(device),
                         th.stack(self.comparisons, axis=0).to(device),
                         beta=self.beta))
    if self.gamma is not None:
      likelihood = (1 - 2 * self.gamma) * likelihood + self.gamma
    return likelihood.prod(axis=-1)

  def log_likelihood(self, preference):
    """"""
    likelihood = (
      compute_likelihood(preference,
                         th.stack(self.items_a, axis=0).to(device),
                         th.stack(self.items_b, axis=0).to(device),
                         th.stack(self.comparisons, axis=0).to(device),
                         beta=self.beta,
                         log=True))
    if self.gamma is not None:
      likelihood = (1 - 2 * self.gamma) * likelihood + self.gamma
    return likelihood.sum(axis=-1)

  def log_likelihood_batch(self, preference):
    return compute_likelihood(preference,
                              th.stack(self.items_a, axis=0).to(device),
                              th.stack(self.items_b, axis=0).to(device),
                              th.stack(self.comparisons, axis=0).to(device),
                              beta=self.beta,
                              eps=self.eps,
                              log=True).sum(axis=0)

  def update(self, a, b, y):
    self.items_a.append(a)
    self.items_b.append(b)
    self.comparisons.append(y)

  def deepcopy(self):
    return {"items_a": self.items_a,
            "items_b": self.items_b,
            "comparisons": self.comparisons}


def uniform_prior(preference):
  if preference.sum() != 1.0:
    return th.tensor(0.0)
  else:
    dirichlet = Dirichlet(th.ones_like(preference))
    return dirichlet.log_prob(preference).exp()


def l2(true, pred):
  # true = data["true_preference"].cpu()
  # steps, pred = unpack(data, key)
  # true = true[np.newaxis, ...].repeat_interleave(len(pred), axis=0)
  return nn.PairwiseDistance(p=2, eps=0)(true, pred)


def projection_unit_simplex(x):
  """Projection onto the unit simplex."""
  s = 1.0
  n_features = x.shape[0]
  u, _ = th.sort(x, descending=True)
  cumsum_u = th.cumsum(u, dim=0)
  ind = th.arange(n_features) + 1
  cond = s / ind + (u - cumsum_u / ind) > 0
  idx = th.count_nonzero(cond)
  return relu(s / idx + (x - cumsum_u[idx - 1] / idx))


def optimize(fun, init, maxiter, learning_rate):

  w = init
  history = [w.clone()]
  for i in range(maxiter):
    _w = nn.Parameter(w)
    obj = fun(_w)
    obj.backward()
    with th.no_grad():
      w = projection_unit_simplex(w + learning_rate * _w.grad)
    history.append(w.clone())
  return history


def pool(nofeedback, size=None, human=False):

  inputs = np.concatenate((nofeedback["input"], nofeedback["input"]), axis=0)
  responses = np.concatenate((nofeedback["response_a"], nofeedback["response_b"]), axis=0)
  goals = np.concatenate((nofeedback["reward_a"], nofeedback["reward_b"]), axis=0)
  if human:
    goals_human = np.concatenate((nofeedback["reward_human_a"], nofeedback["reward_human_b"]), axis=0)
  _, index = np.unique(goals, axis=0, return_index=True)
  inputs = inputs[index]
  responses = responses[index]
  goals = goals[index]
  if human:
    goals_human = goals_human[index]
  index = np.arange(len(inputs))
  pairs = np.array([pair for pair in itertools.product(index, index)])
  pairs = pairs[pairs[:, 0] != pairs[:, 1]]
  if size is not None:
    if len(pairs) >= size:
      pairs = pairs[np.random.choice(np.arange(len(pairs)), size=size, replace=False)]
    else:
      pairs = pairs[np.random.choice(np.arange(len(pairs)), size=size, replace=True)]
  
  responses_a = responses[pairs[:, 0]]
  responses_b = responses[pairs[:, 1]]
  rewards_a = goals[pairs[:, 0]]
  rewards_b = goals[pairs[:, 1]]
  if human:
    rewards_human_a = goals_human[pairs[:, 0]]
    rewards_human_b = goals_human[pairs[:, 1]]
  
  inputs_a = inputs[pairs[:, 0]]
  inputs_b = inputs[pairs[:, 1]]
  # assert (inputs_a != inputs_b).sum() <= 0
  mask = (inputs_a != inputs_b)
  if mask.sum() > 0:
    print(inputs_a[mask])
    print(inputs_b[mask])
    raise AssertionError()

  if human:
    return dict(input=inputs_a,
                response_a=responses_a,
                reward_a=rewards_a,
                reward_human_a=rewards_human_a,
                response_b=responses_b,
                reward_b=rewards_b,
                reward_human_b=rewards_human_b)
  return dict(input=inputs_a,
              response_a=responses_a,
              reward_a=rewards_a,
              response_b=responses_b,
              reward_b=rewards_b)


def pool_v2(nofeedback, size=None):

  inputs = np.concatenate((nofeedback["input"], nofeedback["input"]), axis=0)
  print(len(inputs))
  responses = np.concatenate((nofeedback["response_a"], nofeedback["response_b"]), axis=0)
  goals = np.concatenate((nofeedback["reward_a"], nofeedback["reward_b"]), axis=0)
  _, index = np.unique(goals, axis=0, return_index=True)
  inputs = inputs[index]
  responses = responses[index]
  goals = goals[index]

  unique_inputs = np.unique(inputs, axis=0)
  print(len(inputs))
  data = {"s": [],
          "a1": [],
          "a2": [],
          "r1": [],
          "r2": []}
  for s in unique_inputs:
    mask = (inputs == s)
    index = np.arange(mask.sum())
    pairs = np.array([pair for pair in itertools.product(index, index)])
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    i1 = pairs[:, 0]
    i2 = pairs[:, 1]
    s1 = inputs[mask][i1]
    s2 = inputs[mask][i2]
    if (s1 != s2).sum() > 0:
      raise AssertionError()
    data["s"].append(s1)
    data["a1"].append(responses[mask][i1])
    data["a2"].append(responses[mask][i2])
    data["r1"].append(goals[mask][i1])
    data["r2"].append(goals[mask][i2])
    print(mask.sum())

  inputs = np.concatenate(data["s"])
  responses_a = np.concatenate(data["a1"])
  responses_b = np.concatenate(data["a2"])
  rewards_a = np.concatenate(data["r1"])
  rewards_b = np.concatenate(data["r2"])
  print(inputs.shape)

  index = np.arange(len(inputs))
  if size is not None:
    index = np.random.choice(index, size=size, replace=False)
    # pairs = pairs[np.random.choice(np.arange(len(pairs)), size=size, replace=False)]

  return dict(input=inputs[index],
              response_a=responses_a[index],
              reward_a=rewards_a[index],
              response_b=responses_b[index],
              reward_b=rewards_b[index],
              )


# n_inputs = 32
def get_subset(feedback_dataset, nth, n_inputs=32):
  return feedback_dataset.select(
    np.arange(nth, 
              len(feedback_dataset),
              n_inputs))


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", required=True, type=int)
  parser.add_argument("--task", required=True, type=str)
  parser.add_argument("--rm_names", nargs="+", type=str)
  # parser.add_argument("--nth", required=True, type=int)
  parser.add_argument("--n_rounds", default=10, type=int)
  parser.add_argument("--true_preference", required=True, nargs="+", type=float)
  parser.add_argument("--true_beta", required=True, type=float)
  parser.add_argument("--beta", required=True, type=float)
  parser.add_argument("--n_samples", default=1_000, type=int)
  parser.add_argument("--burnin", default=10_000, type=int)
  parser.add_argument("--thin", default=10, type=int)
  parser.add_argument("--jitter", required=True, type=float)
  parser.add_argument("--alpha", required=True, type=float)
  parser.add_argument("--init", required=False, type=str)
  parser.add_argument("--propose", default="uniform", type=str)
  parser.add_argument("--acquisition", default="volume", type=str)
  parser.add_argument("--inputs", default="0", type=str)
  parser.add_argument("--mode", default="posterior_mean")
  parser.add_argument("--learning_rate", default=None, type=float)
  parser.add_argument("--maxiter", default=1_000, type=int)
  parser.add_argument("--nofeedback_size", default=1_000, type=int)
  parser.add_argument("--distrust", action="store_true")
  parser.add_argument("--F", default=None, type=int)
  parser.add_argument("--E", default=None, type=int)
  parser.add_argument("--gamma", default=None, type=float)
  parser.add_argument("--dims", nargs="+", type=int)
  def none_or_float(a):
    if a == "none":
      return None
    return float(a)
  parser.add_argument("--margin", default=None, type=none_or_float)
  args = parser.parse_args()
  print(args)

  # if args.mode == "likelihood":
  #   if args.learning_rate is None:
  #     raise
  #   if args.maxiter is None:
  #     raise

  if args.distrust:
    if args.F is None:
      raise
    if args.E is None:
      raise

  seed = args.seed
  task = args.task
  rm_names = args.rm_names
  if task in ("summeval", "geval", "summeval+geval"):
    rm_names = ["coherence", "consistency", "fluency", "relevance"]
  else:
    if rm_names is None:
      raise
  true_preference = args.true_preference
  true_beta = args.true_beta
  if np.isinf(true_beta):
    true_beta = None
  true_eps = 0.0
  beta = args.beta
  if np.isinf(beta):
    beta = None
  eps = 0.0
  n_samples = args.n_samples
  burnin = args.burnin
  thin = args.thin
  jitter = args.jitter
  alpha = args.alpha
  hfid = "minhyeonoh"
  wandb.init(entity="personalization",
             project="ACL25",
             config=dict(seed=seed,
                         task=task,
                         rm_names=rm_names,
                         true_preference=true_preference,
                         metropolis_hastings_init=args.init,
                         metropolis_hastings_propose=args.propose,
                         true_beta="inf" if true_beta is None else true_beta,
                         true_eps=true_eps,
                         beta="inf" if beta is None else beta,
                         eps=eps,
                         n_samples=n_samples,
                         burnin=burnin,
                         thin=thin,
                         jitter=jitter,
                         concentration=alpha,
                         acquisition=args.acquisition,
                         inputs=args.inputs,
                         n_rounds=args.n_rounds,
                         mode=args.mode,
                         learning_rate=args.learning_rate,
                         backbone="rewardsincontext",
                         nofeedback_size=args.nofeedback_size,
                         distrust=args.distrust,
                         F=args.F,
                         E=args.E,
                         gamma=args.gamma,
                         margin=args.margin,
                         dims=args.dims,
                         ))
  print(args)

  set_seed(seed)
  true_preference = th.tensor(true_preference)
  if args.dims is not None:
    rm_names = [rm_names[i] for i in args.dims]
    true_preference = true_preference[args.dims]
    true_preference /= true_preference.sum()
    print(true_preference)
  else:
    true_preference /= true_preference.sum()

  # if true_preference.sum() != 1.0:
    # raise
  ndims = len(true_preference)

  # data_dir = f"queries/rewardsincontext/feedback_{task}_{''.join(rm_names)}_{true_preference_str}"
  # nofeedback = load_from_disk(f"{data_dir}/data_19") #
  if task == "summeval":
    nofeedback = load_dataset(f"{hfid}/summeval_paired")
  elif task == "geval":
    nofeedback = load_dataset(f"{hfid}/geval_paired")
  elif task == "summeval+geval":
    nofeedback = load_dataset(f"{hfid}/summeval-geval-paired")
  elif task == "custom":
    data_dir = f"queries/rewardsincontext/feedback_{task}_{''.join(rm_names)}_{true_preference_str}"
    nofeedback = load_from_disk(f"{data_dir}/data_19") #    
  else:
    nofeedback = load_dataset(f"{hfid}/feedback_{task}_{''.join(rm_names)}_271")
  

  nofeedback = nofeedback["train"]
  print(nofeedback)
  if task != "summeval+geval":
    nofeedback.set_format(type="numpy",
                          columns=["input",
                                   "response_a",
                                   "response_b",
                                   "reward_a",
                                   "reward_b"])
  else:
    nofeedback.set_format(type="numpy",
                          columns=["input",
                                   "response_a",
                                   "response_b",
                                   "reward_a",
                                   "reward_b",
                                   "reward_human_a",
                                   "reward_human_b"])

  if task in ("summeval", "geval", "summeval+geval"):
    if args.inputs not in ("all", "dynamic"):
      raise

  if args.dims is not None:
    rewards_a = nofeedback["reward_a"][:, args.dims]
    rewards_b = nofeedback["reward_b"][:, args.dims]
    nofeedback = nofeedback.remove_columns(["reward_a", "reward_b"])
    nofeedback = nofeedback.add_column("reward_a", rewards_a.tolist())
    nofeedback = nofeedback.add_column("reward_b", rewards_b.tolist())
    if task == "summeval+geval":
      rewards_human_a = nofeedback["reward_human_a"][:, args.dims]
      rewards_human_b = nofeedback["reward_human_b"][:, args.dims]
      nofeedback = nofeedback.remove_columns(["reward_human_a", "reward_human_b"])
      nofeedback = nofeedback.add_column("reward_human_a", rewards_a.tolist())
      nofeedback = nofeedback.add_column("reward_human_b", rewards_b.tolist())

  n_inputs = 100 if task in ("summeval", "geval", "summeval+geval") else 32
  if args.inputs not in ("all", "dynamic"):
    nofeedback = get_subset(nofeedback, int(args.inputs), n_inputs=n_inputs)
    nofeedback = Dataset.from_dict(pool(nofeedback, size=args.nofeedback_size))
  if args.inputs == "all":
    nofeedback = Dataset.from_dict(pool_v2(nofeedback, size=args.nofeedback_size))
    assert args.task != "summeval+geval"
  if args.inputs == "dynamic":
    unique_inputs = nofeedback["input"][:n_inputs]
    dynamic_inputs = np.random.choice(n_inputs, 101)
    subsets = []
    dynamic_subsets_index = []
    idx = 0
    # unique_inputs = []
    all_inputs = []
    for i in dynamic_inputs:
      print(i)
      subset = get_subset(nofeedback, i, n_inputs=n_inputs)
      tmp = subset["input"]
      assert len(set(tmp)) == 1
      # unique_inputs.append(tmp[0])
      subset = Dataset.from_dict(pool(subset, size=args.nofeedback_size, human=(task == "summeval+geval")))
      subsets.append(subset)
      dynamic_subsets_index.append(np.arange(idx, idx + len(subset)))
      idx += len(subset)
    nofeedback = concatenate_datasets(subsets)

  if task != "summeval+geval":
    nofeedback.set_format(type="numpy",
                          columns=["input",
                                   "response_a",
                                   "response_b",
                                   "reward_a",
                                   "reward_b"])
  else:
    nofeedback.set_format(type="numpy",
                          columns=["input",
                                   "response_a",
                                   "response_b",
                                   "reward_a",
                                   "reward_b",
                                   "reward_human_a",
                                   "reward_human_b"])

  mask = nofeedback["reward_a"] == nofeedback["reward_b"]
  mask = mask.all(axis=-1)
  assert mask.sum() <= 0

  if task != "summeval+geval":
    nofeedback.set_format(type="torch",
                          columns=["reward_a", 
                                   "reward_b"])
    feedback, diff = (
      generate_feedback(true_preference,
                        nofeedback["reward_a"],
                        nofeedback["reward_b"],
                        beta=true_beta))
    definite_feedback, _ = (
      generate_feedback(true_preference,
                        nofeedback["reward_a"],
                        nofeedback["reward_b"],
                        beta=None))
  else:
    nofeedback.set_format(type="torch",
                          columns=["reward_a", 
                                   "reward_b",
                                   "reward_human_a",
                                   "reward_human_b"])
    feedback, diff = (
      generate_feedback(true_preference,
                        nofeedback["reward_human_a"],
                        nofeedback["reward_human_b"],
                        beta=true_beta))
    definite_feedback, _ = (
      generate_feedback(true_preference,
                        nofeedback["reward_human_a"],
                        nofeedback["reward_human_b"],
                        beta=None))

  if true_beta is None:
    nonzero = diff != 0
    feedback = feedback[nonzero]
    definite_feedback = definite_feedback[nonzero]
    nofeedback = nofeedback.select(*th.nonzero(nonzero, as_tuple=True))
    if args.inputs == "dynamic":
      idx = 0
      for i, subset_index in enumerate(dynamic_subsets_index):
        n = nonzero[subset_index].sum()
        dynamic_subsets_index[i] = np.arange(idx, idx + n)
        idx += n

  # nofeedback.save_to_disk(f"{wandb.run.dir}/nofeedback")
  # shutil.make_archive(f"{wandb.run.dir}/nofeedback", "tar", f"{wandb.run.dir}/nofeedback")
  # if args.inputs != "dynamic":
  #   wandb.save(f"nofeedback.tar")

  print(f"standard deviation of diff is {diff.std()}")
  wandb.log({"standard_deviation_of_diff": diff.std()})
  noise = (definite_feedback != feedback)
  size = args.nofeedback_size
  print(f"noise = {noise.sum()} / {size} = {noise.sum() / size}")
  wandb.log({"comparsion_errors": noise.sum()})

  # intermediate data
  data = {}
  data["target"] = {}
  data["samples"] = {}
  data["estimation"] = {}
  data["score"] = {}
  data["score_isnan"] = {}
  data["max_indices"] = {}
  data["chosen"] = {}
  data["history"] = {}
  if args.inputs == "dynamic":
    data["dynamic_inputs"] = dynamic_inputs
    data["dynamic_subsets_index"] = dynamic_subsets_index

  # start
  target = Target(uniform_prior, beta=beta, eps=eps, gamma=args.gamma)
  data["target"][0] = target.deepcopy() #deepcopy(target)
  proposal = Proposal(jitter=jitter, concentration=alpha, propose=args.propose)
  samples = metropolis_hastings(target,
                                proposal,
                                # th.tensor([1/3, 1/3, 1/3], dtype=th.float, device=device),
                                th.ones(ndims, dtype=th.float, device=device) / ndims,
                                n_samples=n_samples,
                                burnin=burnin,
                                thin=thin)
  data["samples"][0] = samples
  data["estimation"][0] = samples.mean(axis=0)
  t = 1
  while t <= args.n_rounds:
  # for t in range(1, args.n_rounds + 1):
    print(t)
    if args.inputs != "dynamic":
      nofeedbackt = nofeedback
    else:
      assert len(dynamic_inputs) == len(dynamic_subsets_index)
      input_idx = dynamic_inputs[t]
      nofeedbackt = nofeedback.select(dynamic_subsets_index[t])
      print(f"at t={t}, the input is {input_idx}th")
      print(dynamic_subsets_index[t][[0, -1]])

    if args.acquisition.startswith("volume"):
      probs = (
        compute_probs(samples.to(dtype=th.float),
                      nofeedbackt["reward_a"],
                      nofeedbackt["reward_b"],
                      beta=beta))
      probs = probs[..., [0, 2]]
      score = (1 - probs).mean(axis=0).min(axis=-1)[0]
    elif args.acquisition == "entropy":
      curr = data["estimation"][t - 1]
      probs = (
        compute_probs(curr,
                      nofeedbackt["reward_a"],
                      nofeedbackt["reward_b"],
                      beta=beta))
      probs = probs[..., [0, 2]]
      if beta is None:
        raise
      score = -(probs * probs.log()).sum(axis=-1)
    elif args.acquisition == "random":
      score = th.rand(len(nofeedbackt))
    elif args.acquisition == "margin":
      curr = data["estimation"][t - 1]
      probs = (
        compute_probs(curr,
                      nofeedbackt["reward_a"],
                      nofeedbackt["reward_b"],
                      beta=beta))
      probs = probs[..., [0, 2]]
      yhat1 = th.max(probs, axis=-1).values
      yhat2 = 1 - yhat1
      score = -(yhat1 - yhat2)
    else:
      raise

    data["score"][t] = score
    if "variance" not in args.acquisition:
      score_isnan = score.isnan()
      max_indices = th.arange(len(score))[score == th.max(score[~score_isnan])]
      chosen = random.choice(max_indices).item()
      data["score_isnan"][t] = score_isnan
      data["max_indices"][t] = max_indices
    else:
      mask = (score >= th.max(score[~score.isnan()]) - args.margin)
      a1 = nofeedbackt["reward_a"][mask]
      a2 = nofeedbackt["reward_b"][mask]

      print(samples.shape)
      print(a1.shape)
      print(a2.shape)
      lf = compute_likelihood(samples, a1, a2, th.ones(len(a1), dtype=int), beta=beta)
      lf = (1 - 2 * args.gamma) * lf + args.gamma # (1000, U)
      # lf[w,x] = \ell_{\beta,\vw}^\gamma(1|x)
      ls = compute_likelihood(samples, a1, a2, -1 * th.ones(len(a1), dtype=int), beta=beta)
      ls = (1 - 2 * args.gamma) * ls + args.gamma
      # lf[w,x] = \ell_{\beta,\vw}^\gamma(-1|x)

      # muf[w,x,:] = \vw \ell_{\beta,\vw}^\gamma(1|x)]
      #            = samples[w,:] * lf[w,x]
      muf = samples[:, np.newaxis, :] * lf[:, :, np.newaxis] # (1000, 3) (1000, U) -> (1000, U, 3)
      # muf[x,:] = \mu_{t,x}^1
      muf = muf.mean(axis=0) # -> (U, 3)
      mus = samples[:, np.newaxis, :] * ls[:, :, np.newaxis] # (1000, 3) (1000, U) -> (1000, U, 3)
      mus = mus.mean(axis=0) # -> (U, 3)

      # normf[w,x,:] = \vw - \mu_{t,x}^1
      #              = samples[w,:] - muf[x,:]
      # normf[w,x] = |\vw - \mu_{t,x}^1|^2
      normf = samples[:, np.newaxis, :] - muf[np.newaxis, :, :] # (1000, 3) (U, 3) -> (1000, U, 3)
      normf = (normf ** 2).sum(axis=-1) # -> (1000, U)
      norms = samples[:, np.newaxis, :] - mus[np.newaxis, :, :] # (1000, 3) (U, 3) -> (1000, U, 3)
      norms = (norms ** 2).sum(axis=-1) # -> (1000, U)

      # varf[w,x] = |\vw - \mu_{t,x}^1|^2 \ell_{\beta,\vw}^\gamma(-1|x)
      #           = normf[w,x] * lf[w,x]
      varf = normf * lf # (1000, U) (1000, U)
      varf = varf.mean(axis=0) # (U,)
      vars = normf * ls # (1000, U) (1000, U)
      vars = vars.mean(axis=0) # (U,)

      var_score = (varf - vars).abs() # (U,)
      print("!", var_score.shape)
      var_score_isnan = var_score.isnan()
      print(var_score_isnan)
      var_min_indices = th.arange(len(var_score))[var_score == th.min(var_score[~var_score_isnan])]
      print(var_min_indices)
      chosen = random.choice(var_min_indices).item()
      print(chosen)
      print(th.arange(len(score))[mask])
      print(th.arange(len(score))[mask][chosen])
      chosen = th.arange(len(score))[mask][chosen].item()
      print(chosen)

    if args.inputs == "dynamic":
      # print(f"chosen: before={chosen}")
      chosen = dynamic_subsets_index[t][chosen].item()
      # print(f"chosen: after={chosen}")
      # print("---")
      # print(nofeedback["input"][chosen])
      # print("---")
      # print(unique_inputs[input_idx])
      # print(type(nofeedback["input"][chosen]), type(unique_inputs[input_idx]))
      # assert nofeedback["input"][chosen] == unique_inputs[input_idx]

    wandb.log({"t": t, "chosen": chosen})

    if task != "summeval+geval":
      a = nofeedback[chosen]["reward_a"]
      b = nofeedback[chosen]["reward_b"]
    else:
      a = nofeedback[chosen]["reward_human_a"]
      b = nofeedback[chosen]["reward_human_b"]
    comparison, _ = generate_feedback(true_preference,
                                      a[np.newaxis, ...],
                                      b[np.newaxis, ...],
                                      beta=true_beta)
    comparison = comparison[0, ...]
    isnoise = (comparison != definite_feedback[chosen]).float().item()
    print(isnoise)
    wandb.log({"t": t, "isnoise": isnoise})
    print(f"chosen={chosen} a={a} b={b} comparison={comparison.item()} isnoise={isnoise}")
    data["chosen"][t] = dict(index=chosen,
                             a=a,
                             b=b,
                             comparison=comparison.item(),
                             noise=isnoise)

    target.update(a, b, comparison)
    data["target"][t] = target.deepcopy() #deepcopy(target)
    init = th.tensor([1/3, 1/3, 1/3], dtype=th.float, device=device)
    if args.init == "previous":
      init = samples.mean(axis=0)
    samples = metropolis_hastings(target,
                                  proposal,
                                  init,
                                  n_samples=n_samples,
                                  burnin=burnin,
                                  thin=thin)
    data["samples"][t] = samples

    if args.mode == "posterior_mean":
      estimation = samples.mean(axis=0)
      data["estimation"][t] = estimation
    elif (args.mode == "likelihood") and (args.learning_rate is not None):

      def objective_fun(w):
        return target.log_likelihood(w)

      w_init = th.tensor([1/3, 1/3, 1/3])
      w_init = w_init / w_init.sum()
      history = optimize(fun=objective_fun,
                         init=w_init,
                         maxiter=args.maxiter,
                         learning_rate=args.learning_rate)
      history = th.stack(history, axis=0)
      log_likelihood = target.log_likelihood(history)
      estimation = history[th.argmax(log_likelihood)]
      data["history"][t] = history
      data["estimation"][t] = estimation
    elif (args.mode == "likelihood") and (args.learning_rate is None):
      log_likelihood = target.likelihood(samples)
      log_likelihood_max = th.max(log_likelihood)
      max_indices = th.arange(len(log_likelihood))[log_likelihood == log_likelihood_max]
      chosen = random.choice(max_indices).item()
      estimation = samples[chosen]
      data["estimation"][t] = estimation
    else:
      raise

    print(f"estimation={estimation.to(device=device)}")
    wandb.log({"t": t, "l2": l2(true_preference.to(device=device), estimation.to(device=device)).item()})

    if t % 10 == 0:
      with open(f"{wandb.run.dir}/data_{t}.pkl", "wb") as fout:
        pickle.dump(data, fout)
        wandb.save(f"data_{t}.pkl")

    t = t + 1
    # if (t % args.F == 0) and (t <= args.E):
    #   # Update trust coefficients
    #   space = {-beta, beta}^t # (2^t, t)
    #   # variances = 



  with open(f"{wandb.run.dir}/data.pkl", "wb") as fout:
    pickle.dump(data, fout)
    wandb.save("data.pkl")


