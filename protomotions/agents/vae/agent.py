# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Dict, Tuple
import torch
import logging

from torch import Tensor
import math

from lightning.fabric import Fabric
from hydra.utils import instantiate

from protomotions.agents.ppo.utils import bounds_loss
from protomotions.agents.utils.data_utils import swap_and_flatten01
from protomotions.utils.replay_buffer import ReplayBuffer
from protomotions.agents.vae.model import VQVAEModel
from protomotions.agents.common.common import weight_init
from protomotions.envs.base_env.env import BaseEnv
from protomotions.agents.mimic.agent import Mimic

log = logging.getLogger(__name__)


class VQVAE(Mimic):
    # -----------------------------
    # Initialization and Setup
    # -----------------------------
    def setup(self):
        model: VQVAEModel = instantiate(self.config.model)
        model.apply(weight_init)
        actor_optimizer = instantiate(
            self.config.model.config.actor_optimizer,
            params=list(model._actor.parameters()),
        )
        critic_optimizer = instantiate(
            self.config.model.config.critic_optimizer,
            params=list(model._critic.parameters()),
        )
        (
            self.model,
            self.actor_optimizer,
            self.critic_optimizer,
        ) = self.fabric.setup(
            model, actor_optimizer, critic_optimizer
        )
        self.model.mark_forward_method("act")
        self.model.mark_forward_method("get_action_and_value")

    # -----------------------------
    # Experience Buffer and Dataset Processing
    # -----------------------------

    # -----------------------------
    # Reward Calculation
    # -----------------------------

    # -----------------------------
    # Optimization
    # -----------------------------
    
    def calculate_extra_actor_loss(self, batch_dict, dist) -> Tuple[Tensor, Dict]:
        return batch_dict['vq_loss'], {
            "perplexity": batch_dict["perplexity"].detach(),
            "vq_loss": batch_dict["vq_loss"].detach(),
            }    

    # -----------------------------
    # Termination and Logging
    # -----------------------------
    def terminate_early(self):
        self._should_stop = True
