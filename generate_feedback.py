
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch as th

from accelerate import Accelerator
from accelerate.utils import set_seed
from copy import deepcopy
from datasets import load_from_disk
from torch.distributions.dirichlet import Dirichlet
device = "cuda"

SMALL_SIZE = 10
MEDIUM_SIZE = 11
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


def compute_probs(items_a,
                  items_b,
                  preference_vector,
                  beta=None,
                  eps=0):
  """"""

  deterministic = (beta is None)
  beta = beta or 1.0

  ua = (items_a * preference_vector).sum(axis=-1)
  ub = (items_b * preference_vector).sum(axis=-1)
  probs = th.zeros((len(ua), 3), device=device)
  diff = beta * (ua - ub)

  if deterministic:
    tie = th.abs(diff) <= eps
    probs[(diff < 0) & (~tie), 0] = 1.0
    probs[tie, 1] = 1.0
    probs[(diff > 0) & (~tie), 2] = 1.0
  else:
    probs[:, 0] = 1 / (1 + th.exp(+diff))
    probs[:, 2] = 1 / (1 + th.exp(-diff))
  
  return probs


def compute_probs_batch(items_a,
                        items_b,
                        preference_vectors,
                        beta=None,
                        eps=0):
  """"""

  deterministic = (beta is None)
  beta = beta or 1.0
  diff = beta * (items_a - items_b) @ preference_vectors.T
  if deterministic:
    tie = th.abs(diff) <= eps
    return th.stack([(diff < 0) & (~tie),
                     (tie),
                     (diff > 0) & (~tie)], axis=-1).to(dtype=th.float)
  else:
    return th.stack([1 / (1 + th.exp(+diff)),
                     th.zeros_like(diff),
                     1 / (1 + th.exp(-diff))], axis=-1).to(dtype=th.float)


def generate_feedback(items_a,
                      items_b,
                      preference_vector,
                      beta=None,
                      eps=0.0):
  """"""
  probs = compute_probs(items_a,
                        items_b,
                        preference_vector,
                        beta,
                        eps)
  labels = th.multinomial(probs, 1)[:, 0]
  return labels - 1


def compute_likelihood(items_a,
                       items_b,
                       comparisons, 
                       preference,
                       *,
                       beta,
                       eps):
  """"""
  probs = compute_probs(items_a,
                        items_b,
                        preference,
                        beta,
                        eps)
  return probs[th.arange(len(probs)), comparisons + 1]


def metropolis_hastings_v1(target,
                           proposal,
                           init,
                           n_samples,
                           burnin,
                           thin,
                           return_trace=False):
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
  if not return_trace:
    return th.stack(samples[burnin::thin], axis=0)
  else:
    return th.stack(samples[burnin::thin], axis=0), th.stack(samples, axis=0)


def metropolis_hastings_v2(target,
                           proposal,
                           init,
                           n_samples,
                           burnin,
                           thin,
                           return_trace=False):
  """"""

  samples = []
  curr = init
  for i in range(burnin + thin * n_samples - 1):
    while True:
      next = proposal.sample(given=curr)
      if target.prob(next) > 0:
        break
    metropolis_ratio = target.prob(next) / target.prob(curr)
    hastings_ratio = (proposal.prob(curr, given=next) / proposal.prob(next, given=curr))
    if (np.random.uniform(0, 1) < 
        min(1, metropolis_ratio * hastings_ratio)):
      curr = next
    samples.append(curr)
  return th.stack(samples[burnin::thin], axis=0)


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
    else:
      dirichlet = Dirichlet(th.ones_like(given))#(given)
    return dirichlet.log_prob(x).exp()


class Target:

  def __init__(self, prior, beta, eps):

    self.prior = prior
    self.beta = beta
    self.eps = eps

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
    return compute_likelihood(th.stack(self.items_a, axis=0).to(device),
                              th.stack(self.items_b, axis=0).to(device),
                              th.stack(self.comparisons, axis=0).to(device),
                              preference,
                              beta=self.beta,
                              eps=self.eps).prod()

  def update(self, a, b, y):
    self.items_a.append(a)
    self.items_b.append(b)
    self.comparisons.append(y)


def uniform_prior(preference):
  if preference.sum() != 1.0:
    return 0.0
  else:
    dirichlet = Dirichlet(th.ones_like(preference))
    return dirichlet.log_prob(preference).exp()


n_inputs = 32
def get_subset(feedback_dataset, nth):
  return feedback_dataset.select(
    np.arange(nth, 
              len(feedback_dataset),
              n_inputs))

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", required=True, type=int)
  parser.add_argument("--task", required=True, type=str)
  parser.add_argument("--rm_names", required=True, nargs="+", type=str)
  parser.add_argument("--true_preference", required=True, nargs="+", type=float)
  parser.add_argument("--true_beta", required=True, type=float)
  args = parser.parse_args()
  print(args)

  seed = args.seed
  set_seed(seed)

  task = args.task
  rm_names = args.rm_names
  true_preference = th.tensor(args.true_preference)
  if true_preference.sum() != 1.0:
    raise
  true_preference_str = "".join(str(int(w)) for w in (true_preference * 10))

  data = {}
  data["seed"] = seed
  data["task"] = task
  data["rm_names"] = rm_names
  data["true_preference"] = true_preference
  true_beta = args.true_beta
  if np.isinf(true_beta):
    true_beta = None
  true_eps = 0.0
  data["true_beta"] = true_beta
  data["true_eps"] = true_eps

  data_dir = f"feedback_{task}_{''.join(rm_names)}"
  feedback_dataset = load_from_disk(f"{data_dir}/nofeedback")
  feedback_dataset.set_format(type="torch",
                              columns=["reward_a",
                                       "reward_b"])
  labels = (
    generate_feedback(feedback_dataset["reward_a"],
                      feedback_dataset["reward_b"],
                      true_preference,
                      beta=true_beta,
                      eps=true_eps))
  # feedback_dataset = feedback_dataset.remove_columns("comparison")
  feedback_dataset = feedback_dataset.add_column("comparison", labels.cpu().numpy())
  feedback_dataset.set_format(type="torch",
                              columns=["reward_a",
                                       "reward_b",
                                       "comparison"], device=device)
  path = f"{data_dir}/feedback_{true_preference_str}_{args.true_beta}_{true_eps}_{seed}"
  feedback_dataset.save_to_disk(path)
  print(path)
  feedback_dataset = feedback_dataset.select(*th.nonzero(labels, as_tuple=True))
  feedback_dataset = get_subset(feedback_dataset, 0)


