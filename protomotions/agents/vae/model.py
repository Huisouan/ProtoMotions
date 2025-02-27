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

import torch
from torch import nn,distributions
from typing import List
from hydra.utils import instantiate
from protomotions.agents.common.mlp import MLP_WithNorm
from protomotions.agents.common.mlp import MultiHeadedMLP
from protomotions.agents.ppo.model import PPOModel

class VQVAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._encoder: MultiHeadedMLP = instantiate(self.config.encoder)
        self._codebook: nn.Embedding = nn.Embedding(
            self.config.num_embeddings, self.config.embedding_dim
        )
        # 修正初始化参数，原代码存在参数传递错误
        self._codebook.weight.data.uniform_(
            -1.0 / self.config.num_embeddings, 
            1.0 / self.config.embedding_dim
        )
        
    def forward(self, input_dict):
        # 前向传播完整实现
        z = self._encoder(input_dict)  # 获取编码输出
        z_flat = z.view(-1, self.config.embedding_dim)  # 展平特征
        
        # 计算编码与codebook的距离
        distances = (torch.sum(z_flat**2, dim=1, keepdim=True) 
                    + torch.sum(self._codebook.weight**2, dim=1)
                    - 2 * torch.matmul(z_flat, self._codebook.weight.t()))
        
        # 获取最近邻索引
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self._codebook(encoding_indices)  # 量化后的向量
        
        # 直通估计器（Straight-Through Estimator）
        quantized = z + (quantized - z).detach()
        
        # 计算codebook loss和commitment loss
        codebook_loss = torch.mean((quantized.detach() - z)**2)
        commitment_loss = torch.mean((quantized - z.detach())**2)
        vq_loss = codebook_loss + commitment_loss
        return {
            'quantized': quantized.view_as(z),
            'vq_loss': vq_loss,
            'encoding_indices': encoding_indices
        }

class VQVAEDecoder(nn.Module):
    def __init__(self, config, num_out: int):
        super().__init__()
        self.config = config
        self.logstd = nn.Parameter(
            torch.ones(num_out) * config.actor_logstd,
            requires_grad=False,
        )
        self.mu: MultiHeadedMLP = instantiate(self.config.mu_model, num_out=num_out)
    def forward(self, input_dict):
        mu = self.mu(input_dict)
        mu = torch.tanh(mu)
        std = torch.exp(self.logstd)
        dist = distributions.Normal(mu, std)
        return dist

class VQVAEactor(nn.Module):
    def __init__(self, config, num_out: int):
        super().__init__()
        self._encoder: VQVAEEncoder = instantiate(config.encoder)
        
        self._decoder: VQVAEDecoder = instantiate(config.decoder)
    def forward(self, input_dict):
        encoder_out = self._encoder(input_dict)
        decoder_out = self._decoder(input_dict,encoder_out)
        return decoder_out


class VQVAEModel(PPOModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # create networks
        self._actor:VQVAEactor  = instantiate(
            self.config.actor,
        )
        self._critic: MultiHeadedMLP = instantiate(
            self.config.critic,
        )
        
    def get_action_and_value(self, input_dict: dict):
        dist = self._actor(input_dict)
        action = dist.sample()
        value = self._critic(input_dict).flatten()

        logstd = self._actor.logstd
        std = torch.exp(logstd)

        neglogp = self.neglogp(action, dist.mean, std, logstd)
        return action, neglogp, value.flatten()

    def act(self, input_dict: dict, mean: bool = True) -> torch.Tensor:
        dist = self._actor(input_dict)
        if mean:
            return dist.mean
        return dist.sample()

    @staticmethod
    def neglogp(x, mean, std, logstd):
        return (
            0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1]
            + logstd.sum(dim=-1)
        )
