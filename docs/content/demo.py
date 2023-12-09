#%%
#!/usr/bin/env python
# coding: utf-8

# <a target="_blank" href="https://colab.research.google.com/github/ai-safety-foundation/sparse_autoencoder/blob/main/docs/content/demo.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# # Quick Start Training Demo
# 
# This is a quick start demo to get training a SAE right away. All you need to do is choose a few
# hyperparameters (like the model to train on), and then set it off.
# By default it replicates Neel Nanda's
# [comment on the Anthropic dictionary learning
# paper](https://transformer-circuits.pub/2023/monosemantic-features/index.html#comment-nanda).

# ## Setup

# ### Imports

# In[5]:

# Check if we're in Colab
try:
    import google.colab  # noqa: F401 # type: ignore

    in_colab = True
except ImportError:
    in_colab = False

#  Install if in Colab
if in_colab:
    get_ipython().run_line_magic('pip', 'install sparse_autoencoder transformer_lens transformers wandb')

# Otherwise enable hot reloading in dev mode
if not in_colab:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')


# In[6]:


import os

from sparse_autoencoder import (
    ActivationResamplerHyperparameters,
    Hyperparameters,
    LossHyperparameters,
    Method,
    PipelineHyperparameters,
    OptimizerHyperparameters,
    Parameter,
    SourceDataHyperparameters,
    SourceModelHyperparameters,
    sweep,
    SweepConfig,
)
import transformer_lens
from sparse_autoencoder.train.sweep import *
import wandb


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_NOTEBOOK_NAME"] = "demo.ipynb"

#%%

lm_name = "gpt2-small"
model_config = transformer_lens.loading_from_pretrained.get_pretrained_model_config(lm_name)

#%%

# ### Hyperparameters
# Customize any hyperparameters you want below (by default we're sweeping over l1 coefficient and
# learning rate):

# In[7]:

lm_name = "gpt2-small"
batch_size = 4096
sweep_config = SweepConfig(
    parameters=Hyperparameters(
        activation_resampler=ActivationResamplerHyperparameters(
            threshold_is_dead_portion_fires=Parameter(1e-6),
        ),
        loss=LossHyperparameters(
            l1_coefficient=Parameter(max=1e-2, min=4e-3),
        ),
        optimizer=OptimizerHyperparameters(
            lr=Parameter(1.2 * 1e-3),
        ),
        source_model=SourceModelHyperparameters(
            name=Parameter(lm_name),
            hook_layer=Parameter(8),
            hook_site=Parameter("mlp_out"),
            hook_dimension=Parameter(model_config.d_model),
        ),
        source_data=SourceDataHyperparameters(
            dataset_path=Parameter("NeelNanda/openwebtext-tokenized-9b"),
        ),
        pipeline = PipelineHyperparameters(
            train_batch_size = Parameter(batch_size),
            validation_frequency = Parameter(batch_size * 100),
        )
    ),
    method=Method.RANDOM,
)
sweep_config

# ### Run the sweep


#%%

hyperparameters = sweep_config.to_dict()["parameters"]

def simplify_dict(d):
    if isinstance(d, dict):
        if 'parameters' in d:
            # Replace the entire dict with the 'value' of its children
            return {k: (v['value'] if 'value' in v else v) for k, v in d['parameters'].items()}
        else:
            # Recursively apply this simplification to all dict items
            return {k: simplify_dict(v) for k, v in d.items()}
    return d

hyperparameters = simplify_dict(hyperparameters)
hyperparameters['random_seed'] = hyperparameters['random_seed']['value'] # Dunno why this is not caught but whatever
hyperparameters['loss']['l1_coefficient'] = 0.01 # Not min/max... #, 'min': 0.004}},

#%%

# Setup the device for training
device = get_device()

# Set up the source model
source_model = setup_source_model(hyperparameters)

# Set up the autoencoder
autoencoder = setup_autoencoder(hyperparameters, device)

# Set up the loss function
loss_function = setup_loss_function(hyperparameters)

# Set up the optimizer
optimizer = setup_optimizer(autoencoder, hyperparameters)

# Set up the activation resampler
activation_resampler = setup_activation_resampler(hyperparameters)

# Set up the source data
source_data = setup_source_data(hyperparameters)

# The last run we actually did the last part too

#%%

# # Run the training pipeline
# run_training_pipeline(
#     hyperparameters=hyperparameters,
#     source_model=source_model,
#     autoencoder=autoencoder,
#     loss=loss_function,
#     optimizer=optimizer,
#     activation_resampler=activation_resampler,
#     source_data=source_data,
#     run_name="Hello",
# )

#%%

def my_func():
    try:
        run_training_pipeline(
            hyperparameters=hyperparameters,
            source_model=source_model,
            autoencoder=autoencoder,
            loss=loss_function,
            optimizer=optimizer,
            activation_resampler=activation_resampler,
            source_data=source_data,
            run_name="Hello",
        )
    except Exception as e:
        print(e, "was the error", flush=True)
    wandb.finish()

%prun my_func()

#%%

%load_ext line_profiler
import transformer_lens

#%%

# pip install line_profiler
def my_func():
    try:
        lm_name = "gpt2-small"
        transformer_lens.HookedTransformer.from_pretrained(lm_name)
        assert False
    except Exception as e:
        print(e)

%prun my_func()

#%%
