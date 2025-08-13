
import numpy as np
import os
import pandas as pd
import torch
import torch as th

from accelerate import Accelerator
from accelerate.utils import gather_object, set_seed
from dataclasses import dataclass, field
from datasets import concatenate_datasets, Dataset, load_dataset
from multi_reward_models import RewardModels
from peft import PeftModel
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          DataCollatorWithPadding,
                          HfArgumentParser)
from typing import Optional
from utils import (get_clean_data, 
                   Instructions_n,
                   load_main_tokenizer,
                   Instructions_summary_n,
                   save_configs,
                   map_rewards_from_preference)
tqdm.pandas()

# define paths for two datasets
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'

@dataclass
class ScriptArguments:
  peft_name: Optional[str] = field(default='', metadata={'help': "path of the peft files"})
  num_prefer_points: Optional[int] = field(default=10)
  log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
  save_directory: Optional[str] = field(default='./logs_trl_eval/')
  wandb_name: Optional[str] = field(default='test', metadata={"help": "Name for this experiment"})
  reward_names:Optional[str] = field(default='harmless,helpful') 
  base_model_name: Optional[str] = field(default='meta-llama/Llama-2-7b-hf', metadata={"help": "local path to the base model or the huggingface id"})
  reward_stats_path: Optional[str] = field(default='')
  exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})


accelerator = Accelerator()
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
base_model_name = script_args.base_model_name
tokenier_name = script_args.base_model_name
reward_stats_path = script_args.reward_stats_path if len(script_args.reward_stats_path) else None
accelerator.print('base model: ', base_model_name)

peft_name = script_args.peft_name
reward_names = [x.strip() for x in script_args.reward_names.split(',')]
accelerator.print(reward_names)
reward_path_tokenizer_dict = {
    'harmless': ['Ray2333/gpt2-large-harmless-reward_model'],
    'helpful': ['Ray2333/gpt2-large-helpful-reward_model'],
    'deberta': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary': ['Tristan/gpt2_reward_summarization'],
    'faithful':['CogComp/bart-faithful-summary-detector'],
    'humor': ['mohameddhiab/humor-no-humor'],
}
reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
  if name not in reward_path_tokenizer_dict.keys():
    raise NotImplementedError
  reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
  rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])
    

save_info = {
    'base_model_name': base_model_name,
    'peft_name': peft_name,
    'tokenier_name': tokenier_name
}
for i in range(len(reward_model_path_list)):
    save_info['reward_peft_path{}'.format(i+1)] = reward_model_path_list[i]
save_configs(save_info, os.path.join(script_args.save_directory, script_args.wandb_name))


process_id = Accelerator().local_process_index 
gpu_id = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

set_seed(8888)
tokenizer = load_main_tokenizer(tokenier_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    torch_dtype=torch.bfloat16,  # fast inference
    device_map=gpu_id, 
)
############# very important for padding
model.resize_token_embeddings(len(tokenizer))
if len(peft_name) > 0:
  model = PeftModel.from_pretrained(model, peft_name)
if hasattr(model, 'merge_and_unload'):
  model = model.merge_and_unload()

# load reward model
# do not normalization for evaluation
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id, reward_stats_path) 
num_rewards = len(reward_model_path_list)
instructions = Instructions_n(num_rewards) if exp_type == 'assistant' else Instructions_summary_n(num_rewards)

generation_kwargs = {
    "max_new_tokens": 128 if exp_type == 'assistant' else 48, 
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.9, 
    "do_sample": True,
}

### for evaluation
accelerator.print('evaluation........')
tokenizer.padding_side = "left"

## preference list
if reward_models.num_rewards == 2:
  N = script_args.num_prefer_points
  preferences = np.zeros((N+1, 2))
  preferences[:, 0] = np.arange(0,1 + 1/N, 1/N)
  preferences[:, 1] = 1 - preferences[:, 0]
  preferences = np.round(preferences, 1)
elif reward_models.num_rewards == 3:
  preferences = np.array([
      [0.0, 0.0, 1.0],
      [0.0, 1.0, 0.0],
      [0.1, 0.1, 0.8],
      [0.1, 0.8, 0.1],
      [0.2, 0.2, 0.6],
      [0.2, 0.4, 0.4],
      [0.2, 0.6, 0.2],
      [0.33, 0.33, 0.33],
      [0.4, 0.4, 0.2],
      [0.4, 0.2, 0.4], 
      [0.6, 0.2, 0.2],
      [0.8, 0.1, 0.1], 
      [1.0, 0.0, 0.0], 
      ])
else: 
  raise NotImplementedError


# using Gaussian rewards as reference
rewards_reference_list = [np.random.randn(50000) for _ in range(len(preferences[0]))]
def evaluate(data,
             model, 
             rm,
             tokenizer, 
             instructions):
 
    accelerator.print(f"Size of the validation set: {len(data)}")
    valid_batch_size = 1 # should be 1 due to the padding
    data = data.remove_columns('input_ids')
    data = data.rename_column('prompt_with_score_ids', 'input_ids')
    data = data.remove_columns(['prompt', 'response', 'query', 'score1', 'score2', 'prompt_with_score'])
    for key in ['key', 'text']:
      if key in data.column_names:
        data = data.remove_columns(key)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(data, batch_size=valid_batch_size, collate_fn=data_collator)
    dataloader = accelerator.prepare(dataloader)

    def compute_reward(inputs, responses):
      inputs_and_responses = list(zip(inputs, responses))
      if hasattr(instructions, 'get_post'):
        return rm.get_reward_model_scores(inputs_and_responses, instructions.get_post)
      else:
        return rm.get_reward_model_scores(inputs_and_responses)

    outputs = []
    for batch in tqdm(dataloader,
                      desc=f"?",
                      disable=(not accelerator.is_local_main_process)):
      input = batch["input_ids"]
      with th.no_grad():
        output = model.generate(input, **generation_kwargs)

      input = tokenizer.batch_decode(input)
      output = tokenizer.batch_decode(output)
      _, output = get_clean_data(output, input)
      outputs.extend(
        accelerator.gather_for_metrics(output))

    with accelerator.split_between_processes(outputs) as local_outputs:
      inputs = [instructions.get_input(output) for output in local_outputs]
      responses = [instructions.get_response(output) for output in local_outputs]
      rewards = compute_reward(inputs, responses)

    rm_inputs = accelerator.gather_for_metrics(inputs)
    rm_responses = accelerator.gather_for_metrics(responses)
    rm_outputs = []
    for i in range(num_rewards):
      rm_outputs.append(
        accelerator.gather_for_metrics(rewards[i]))
    rm_outputs = np.array(rm_outputs).T

    assert len(rm_inputs) == len(outputs)
    assert len(rm_inputs) == len(rm_responses)
    for i in range(num_rewards):
      assert len(rm_inputs) == len(rm_outputs[:, i])

    return rm_inputs, rm_responses, rm_outputs


rm_tokenizers = []
for i in range(num_rewards):
  rm_tokenizers.append(AutoTokenizer.from_pretrained(rm_tokenizer_path_list[i]))



def preprocess(dataset, tokenizer, rm_tokenizers, size=None):
  if size is not None:
    dataset = dataset.select(range(size))
  dataset = dataset.select(range(0, len(dataset), 4))
  n = len(rm_tokenizers)

  def tokenize(sample, reject=False):
    if not reject:
      sample['text'] = sample['chosen'] 
    else:
      sample['text'] = sample['rejected'] 
    split_text = sample['text'].split('\n\nAssistant:')
    sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
    sample['response'] = split_text[-1].strip()
    sample["input_ids"] = tokenizer.encode(sample["text"])
    sample["query"] = tokenizer.decode(sample["input_ids"])
    for i in range(n):
      if type(rm_tokenizers[i]) != str:
        sample['reward_ids{}'.format(1+i)] = rm_tokenizers[i].encode(sample['text'])
    return sample

  with accelerator.main_process_first():
    dataset = dataset.map(tokenize, batched=False, fn_kwargs={"reject": False})

  accelerator.wait_for_everyone()
  with accelerator.main_process_first():
    dataset = dataset.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['rejected', 'chosen']
    for i in range(n):
      if type(rm_tokenizers[i]) != str:
        dataset = dataset.filter(lambda x: len(x['reward_ids{}'.format(1+i)]) <= 512 and len(x['reward_ids{}'.format(1+i)]) >= 8)
        remove_columns.append('reward_ids{}'.format(1+i))
  
  dataset = dataset.remove_columns(remove_columns)
  accelerator.wait_for_everyone()
  return dataset


def preprocess_summary(dataset, tokenizer, rm_tokenizers, size=None):

  accelerator.wait_for_everyone()
  with accelerator.main_process_first():
    dataset = dataset.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200 and x['info']["id"] is not None, batched=False, num_proc=30)

  # Remove duplicated inputs
  # accelerator.wait_for_everyone()
  # with accelerator.main_process_first():
    dataset = dataset.filter(lambda x: x['info']["id"] is not None)
    initial_list = dataset.map(lambda x: {"id": x['info']["id"]})
    _ , unique_indices = np.unique(initial_list["id"], return_index=True, axis=0)
    dataset = dataset.select(unique_indices.tolist())

  if size is not None:
    dataset = dataset.select(range(size))
  dataset = dataset.select(range(0, min(len(dataset),2000)))
  n = len(rm_tokenizers)

  instructions = Instructions_summary_n(n)
  def tokenize(sample, chosen=True):
    info_post = sample["info"]["post"].replace("\n", " ")
    prompt_summary = instructions.prompt_input_noscore(info_post)
    sample["prompt"] = prompt_summary
    if chosen:
      choice = sample["choice"] # select the best summary
    else:
      choice = 0 if sample["choice"] != 0 else 1

    sample["response"] = sample["summaries"][choice]["text"].replace("\n", " ").strip()
    sample["input_ids"] = tokenizer.encode(prompt_summary + sample["response"])
    sample["query"] = tokenizer.decode(sample["input_ids"])
    for i in range(n):
      if type(rm_tokenizers[i]) != str:
        sample['reward_ids{}'.format(1+i)] = rm_tokenizers[i].encode(sample["query"])
    return sample

  with accelerator.main_process_first():
    dataset = dataset.map(tokenize, batched=False)

  accelerator.wait_for_everyone()
  with accelerator.main_process_first():
    dataset = dataset.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['info', 'summaries', 'choice', 'worker', 'batch', 'split', 'extra']
    for i in range(n):
      if type(rm_tokenizers[i]) != str:
        dataset = dataset.filter(lambda x: len(x['reward_ids{}'.format(1+i)]) <= 512 and len(x['reward_ids{}'.format(1+i)]) >= 8)
        remove_columns.append('reward_ids{}'.format(1+i))
  
  dataset = dataset.remove_columns(remove_columns)
  accelerator.wait_for_everyone()
  return dataset


def augment(dataset, tokenizer, rm_tokenizers, targets):

  assert len(dataset) == len(targets)
  n = len(rm_tokenizers)
  instructions = Instructions_n(n)
  for i in range(n):
    dataset = dataset.add_column('score{}'.format(i+1), targets[:, i])

  def add_score(sample):
    sample['prompt_with_score'] = sample['prompt'].rstrip('\n\nAssistant:') + ' ' 
    for i in range(n):
      sample['prompt_with_score'] += instructions.score_splits[i] + ' ' + str(sample['score{}'.format(i+1)]) + ' '  
    sample['prompt_with_score'] += '\n\nAssistant:'
    sample['prompt_with_score_ids'] = tokenizer.encode(sample['prompt_with_score'])
    sample["input_ids"] = tokenizer.encode(sample["prompt_with_score"] + ' ' + sample['response'])
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample

  with accelerator.main_process_first():
    dataset = dataset.map(add_score, batched=False)
  dataset.set_format(type="torch")
  return dataset


def augment_summary(dataset, tokenizer, rm_tokenizers, targets):

  assert len(dataset) == len(targets)
  n = len(rm_tokenizers)
  instructions = Instructions_summary_n(n)
  for i in range(n):
    dataset = dataset.add_column('score{}'.format(i+1), targets[:, i])

  def add_score(sample):
    sample['prompt_with_score'] = sample['prompt'].rstrip('### Response:') + ' ' 
    for i in range(n):
      sample['prompt_with_score'] += instructions.score_splits[i] + ' ' + str(round(sample['score{}'.format(i+1)], 1)) + ' '
    sample['prompt_with_score'] += '### Response:'
    sample['prompt_with_score_ids'] = tokenizer.encode(sample['prompt_with_score'])
    sample["input_ids"] = tokenizer.encode(sample["prompt_with_score"] + ' ' + sample['response'])
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample

  with accelerator.main_process_first():
    dataset = dataset.map(add_score, batched=False)
  dataset.set_format(type="torch")
  return dataset


if exp_type == "assistant":
  dataset = load_dataset(hhrlhf_dataset_path, split="test")
  dataset = preprocess(dataset, tokenizer, rm_tokenizers)
else:
  dataset = load_dataset(summary_dataset_path, 'comparisons')
  dataset = dataset["validation"]
  dataset = preprocess_summary(dataset, tokenizer, rm_tokenizers)

n_inputs = 32
dataset = dataset.select(
  np.random.choice(len(dataset),
                   size=n_inputs, replace=False))
accelerator.print(dataset)

def choice_from_simplex(dim, size=()):
  def choice():
    k = np.random.exponential(scale=1.0, size=dim)
    return k / sum(k)
  res = np.array([choice() for _ in range(int(np.prod(size)))])
  return res.reshape((*size, dim))

def random_targets(dim, n):
  targets = []
  for w in choice_from_simplex(dim, size=(n,)):
    targets.append(map_rewards_from_preference(rewards_reference_list, w, method='l2').reshape(-1))
  return np.stack(targets, axis=0)

def compute_feedback(rewards_a, 
                     rewards_b, 
                     preference, 
                     deterministic=True, 
                     indifference_thres=None):

  assert rewards_a.shape == rewards_b.shape
  assert len(preference.shape) == 1
  assert rewards_a.shape[-1] == preference.shape[0]

  ua = (rewards_a * preference).sum(axis=-1)
  ub = (rewards_b * preference).sum(axis=-1)
  diff = ua - ub
  labels = (diff > 0 if deterministic else 
            th.bernoulli(th.sigmoid(
              th.from_numpy(diff))).numpy()).astype(np.float32)
  if indifference_thres is not None:
    labels[np.abs(diff) <= indifference_thres] = 0.5
  return labels


true_preference = np.array([0.6, 0.3, 0.1]) #choice_from_simplex(dim=num_rewards)
# true_preference = np.array([0.1, 0.9]) #choice_from_simplex(dim=num_rewards)
true_preference_str = "".join(str(int(w)) for w in (true_preference * 10))
accelerator.print(true_preference)
deterministic = True
indifference_thres = None

budget = 16
n_rounds = 20

opening_inputs = concatenate_datasets([dataset for _ in range(budget)])
feedback_dataset = Dataset.from_dict({})
def get_subset(nth):
  return feedback_dataset[np.arange(nth, 
                                    len(feedback_dataset),
                                    n_inputs)]

preference = choice_from_simplex(dim=num_rewards, size=(1,))
preference = np.repeat(preference, n_inputs, axis=0)
preference = nn.Parameter(th.tensor(preference))

for t in range(n_rounds):

  inputs = {}
  responses = {}
  rewards = {}
  for ab in ["a", "b"]:
    # Augment inputs with random targets in reward space.
    if exp_type == "assistant":
      data = augment(opening_inputs, 
                     tokenizer, 
                     rm_tokenizers, 
                     random_targets(dim=num_rewards, 
                                    n=len(opening_inputs)))
    else:
      data = augment_summary(opening_inputs,
                             tokenizer,
                             rm_tokenizers,
                             random_targets(dim=num_rewards,
                                            n=len(opening_inputs)))
    # Generate reponses and compute rewards thereof.
    (inputs[ab],
     responses[ab],
     rewards[ab]) = evaluate(data, 
                             model, 
                             reward_models, 
                             tokenizer,
                             instructions)
  for s1, s2 in zip(inputs["a"], inputs["b"]):
    assert s1 == s2

  # Generate comparisons.
  comparisons = compute_feedback(rewards["a"],
                                 rewards["b"],
                                 true_preference,
                                 deterministic,
                                 indifference_thres)
  # Add to dataset.
  new_feedback = dict(input=inputs["a"],
                      response_a=responses["a"],
                      reward_a=rewards["a"],
                      response_b=responses["b"],
                      reward_b=rewards["b"],
                      comparison=comparisons)
  feedback_dataset = concatenate_datasets(
    [feedback_dataset, Dataset.from_dict(new_feedback)])
  accelerator.print(f"Round {t}")
  accelerator.print(feedback_dataset)

  if process_id == 0:
    feedback_dataset.save_to_disk(f"queries/rewardsincontext/feedback_{exp_type}_{''.join(reward_names)}_{true_preference_str}/data_{t}")
    # feedback_dataset.save_to_disk(f"queries/rewardedsoups/feedback_{exp_type}_{''.join(reward_names)}_{true_preference_str}/data_{t}")

    # import pickle
    # with open("inputs", "wb") as fout:
    #   pickle.dump(inputs, fout)
    ...
    # evaluation_result = {
    #     'prompt': all_full_prompts,
    #     'response': all_full_responses,
    # }
    # for i in range(num_rewards):
    #   evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
    #   evaluation_result['desired_score{}'.format(i+1)] = all_desired_rewards[i]

    #   print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))
    #   print('total average desired score {}: {}'.format(i+1, np.mean(evaluation_result['desired_score{}'.format(i+1)])))

    # dataframe = pd.DataFrame(evaluation_result)
    # if len(preference) == 3:
    #   dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}_{}.csv'.format(preference[0], preference[1], preference[2])))
    # else:
    #   dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}.csv'.format(preference[0], preference[1])))

# CUDA_VISIBLE_DEVICES=7,6,5,4 accelerate launch --main_process_port=29501 gen.py --peft_name logs_trl/ric_summary_summarydebertafaithful_offline20000_onlineiter2/model_iter2 --reward_names 'summary,deberta,faithful' --exp_type 'summary'

