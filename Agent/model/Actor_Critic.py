import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from Agent.model.GCN import GCN
from torch.distributions import MultivariateNormal, Normal


class Actor(nn.Module):
    def __init__(self, settings, config):
        super(Actor, self).__init__()
        self.settings = settings
        self.config = config
        self.action_std = config.init_action_std if hasattr(config, 'init_action_std') else 0.1
        self.l1_input_dim = config.state_dim
        if self.config.use_history_state or self.config.use_history_action:
            self.l1_input_dim += config.gru_hidden_size
        if self.config.use_topology_info:
            self.l1_input_dim += config.gcn_hidden_size
        self.l1 = nn.Linear(self.l1_input_dim, 512)
        self.l2 = nn.Linear(512, 512)
        # split action header
        self.head_thermal_gen_p = nn.Linear(512, config.more_detail_action_dim['thermal_gen_p'])
        self.head_renewable_gen_p = nn.Linear(512, config.more_detail_action_dim['renewable_gen_p'])
        self.head_adjld_p = nn.Linear(512, config.more_detail_action_dim['adjld_p'])
        self.head_stoenergy_p = nn.Linear(512, config.more_detail_action_dim['stoenergy_p'])
        # use gru to process history state sequence
        if self.config.use_history_state or self.config.use_history_action:
            self.gru_input_size = self.config.state_dim * self.config.use_history_state + \
                                  self.config.action_dim * self.config.use_history_action
            self.gru = nn.GRU(
                input_size=self.gru_input_size,
                hidden_size=self.config.gru_hidden_size,
                num_layers=self.config.gru_num_layers,
                batch_first=True
            )
            self.state_mask = nn.Parameter(
                torch.cat([torch.ones(self.config.state_dim, dtype=torch.float32),
                           torch.zeros(config.gru_hidden_size, dtype=torch.float32)]).unsqueeze(0),
                requires_grad=True,
            )
            self.end_of_token = nn.Parameter(
                torch.zeros(self.gru_input_size, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                requires_grad=True,
            )
        # use gcn to process topology structure
        if self.config.use_topology_info:
            self.gcn = GCN(
                nfeat=self.config.feature_num,
                nhid=self.config.gcn_hidden_size,
                nclass=self.config.gcn_hidden_size,
                dropout=self.config.gcn_dropout,
            )
        # use switch header to control thermal on/off
        if self.config.restrict_thermal_on_off:
            self.head_switch = nn.Linear(512, 2 * config.more_detail_action_dim['thermal_gen_p'])
            self.type_vector = torch.tensor(np.array([0.0, 1.0]), dtype=torch.float32).to(config.device)
        if self.config.init_model:
            self.apply(init_model)

    def forward(self, x, h_x=None, h_a=None, feat=None, adj=None, action_low=None, action_high=None, sample=False):
        if self.config.use_history_state or self.config.use_history_action:
            """
            actor stage: 
                h_x: (1, seq_len, state_dim) -> (1, seq_len + 1, state_dim)
                h_a: (1, seq_len, action_dim) -> (1, seq_len + 1, action_dim)
                h_input: (1, seq_len + 1, state_dim + action_dim)
                hidden: (num_layers, 1, gru_hidden_size)
                output: (1, seq_len, gru_hidden_size)
            learner stage:
                h_x: (batch_size, seq_len, state_dim) -> (batch_size, seq_len + 1, state_dim)
                h_a: (batch_size, seq_len, action_dim) -> (batch_size, seq_len + 1, action_dim)
                h_input: (batch_size, seq_len + 1, state_dim + action_dim)
                hidden: (num_layers, batch_size, gru_hidden_size)
                output: (batch_size, seq_len, state_dim)
            """
            hidden = torch.zeros(self.config.gru_num_layers, x.shape[0], self.config.gru_hidden_size).to(
                self.config.device)
            h_input = torch.cat([h_x, h_a], dim=-1) if self.config.use_history_state and self.config.use_history_action \
                else (h_x if self.config.use_history_state else h_a)
            h_input = torch.cat([h_input, self.end_of_token.repeat(h_input.shape[0], 1, 1)], dim=1)
            output, hidden = self.gru(h_input, hidden)
            x = torch.cat([x, output[:, -1, :]], dim=-1) * self.state_mask
        if self.config.use_topology_info:
            """
            actor stage: 
                feat: (1, num_node, num_feature)
                adj: (1, num_node, num_node)
                output: (1, gcn_hidden_size)
            learner stage:
                feat: (batch_size, num_node, num_feature)
                adj: (batch_size, num_node, num_node)   
                output: (batch_size, gcn_hidden_size)
            """
            output1 = self.gcn(feat, adj)
            x = torch.cat([x, output1], dim=-1)
        x = F.relu(self.l1(x)) if self.config.active_function == 'relu' else F.tanh(self.l1(x))
        x = F.relu(self.l2(x)) if self.config.active_function == 'relu' else F.tanh(self.l2(x))
        thermal_gen_p = self.head_thermal_gen_p(x)
        renewable_gen_p = self.head_renewable_gen_p(x)
        adjld_p = self.head_adjld_p(x)
        stoenergy_p = self.head_stoenergy_p(x)
        # switch header output selection for applying action or not
        if self.config.restrict_thermal_on_off:
            thermal_switch = self.head_switch(x).reshape(-1, self.config.more_detail_action_dim['thermal_gen_p'], 2)
            one_hot = nn.functional.gumbel_softmax(thermal_switch, hard=True, dim=-1)
            thermal_switch = (one_hot * self.type_vector).sum(-1)
        # limit action to [0, 1] and reflect yo valid action space
        if self.config.reflect_actionspace and action_low is not None and action_high is not None:
            thermal_gen_p = F.tanh(thermal_gen_p)
            renewable_gen_p = F.tanh(renewable_gen_p)
            adjld_p = F.tanh(adjld_p)
            stoenergy_p = F.tanh(stoenergy_p)

        return self.combine_action(thermal_gen_p, renewable_gen_p,
                                   adjld_p, stoenergy_p,
                                   action_low, action_high, sample,
                                   thermal_switch if self.config.restrict_thermal_on_off
                                   else torch.ones_like(thermal_gen_p).to(self.config.device))

    def combine_action(self, thermal_gen_p, renewable_gen_p, adjld_p, stoenergy_p,
                       action_low, action_high, sample, thermal_switch):
        thermal_gen_p_i, renewable_gen_p_i = 0, 0
        gen_p = torch.zeros([thermal_gen_p.shape[0], self.settings.gen_num]).to(self.config.device)
        gen_switch = torch.ones([thermal_gen_p.shape[0], self.settings.gen_num]).to(self.config.device)
        for i, gen_type in enumerate(self.settings.gen_type):
            if gen_type == 1:
                gen_p[:, i] = thermal_gen_p[:, thermal_gen_p_i]
                gen_switch[:, i] = thermal_switch[:, thermal_gen_p_i]
                thermal_gen_p_i += 1
            elif gen_type == 5:
                gen_p[:, i] = renewable_gen_p[:, renewable_gen_p_i]
                renewable_gen_p_i += 1
        gen_v = torch.zeros_like(gen_p).to(self.config.device)
        # add random noise to sample action
        # only sample at actor stage
        if sample:
            gen_p += torch.from_numpy(np.random.normal(0, self.action_std, size=gen_p.shape[0])).to(self.config.device)
            adjld_p += torch.from_numpy(np.random.normal(0, self.action_std, size=adjld_p.shape[0])).to(self.config.device)
            stoenergy_p += torch.from_numpy(np.random.normal(0, self.action_std, size=stoenergy_p.shape[0])).to(self.config.device)
        # cat action by rules
        action_output = torch.cat([gen_p, gen_v, adjld_p, stoenergy_p], dim=-1)
        action_switch = torch.cat([gen_switch,
                                   torch.ones_like(gen_v).to(self.config.device),
                                   torch.ones_like(adjld_p).to(self.config.device),
                                   torch.ones_like(stoenergy_p).to(self.config.device)], dim=-1)
        # reflect action output to valid action space
        if self.config.reflect_actionspace:
            action_output = (action_high - action_low) * action_output / 2 + (action_high + action_low) / 2
        if self.config.restrict_thermal_on_off:
            action_output *= action_switch

        return action_output


class Critic(nn.Module):
    def __init__(self, settings, config):
        super(Critic, self).__init__()
        self.settings = settings
        self.config = config
        self.l1_input_dim = config.state_dim + config.action_dim
        if self.config.use_history_state or self.config.use_history_action:
            self.l1_input_dim += config.gru_hidden_size
        if self.config.use_topology_info:
            self.l1_input_dim += config.gcn_hidden_size
        self.l1 = nn.Linear(self.l1_input_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 1)
        # use gru to process history state sequence
        if self.config.use_history_state or self.config.use_history_action:
            self.gru_input_size = self.config.state_dim * self.config.use_history_state + \
                                  self.config.action_dim * self.config.use_history_action
            self.gru = nn.GRU(
                input_size=self.gru_input_size,
                hidden_size=self.config.gru_hidden_size,
                num_layers=self.config.gru_num_layers,
                batch_first=True
            )
            self.state_mask = nn.Parameter(
                torch.cat([torch.ones(self.config.state_dim, dtype=torch.float32),
                           torch.zeros(config.gru_hidden_size, dtype=torch.float32)]),
                requires_grad=True,
            )
            self.end_of_token = nn.Parameter(
                torch.zeros(self.gru_input_size, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                requires_grad=True,
            )
        # use gcn to process topology structure
        if self.config.use_topology_info:
            self.gcn = GCN(
                nfeat=self.config.feature_num,
                nhid=self.config.gcn_hidden_size,
                nclass=self.config.gcn_hidden_size,
                dropout=self.config.gcn_dropout,
            )
        if self.config.init_model:
            self.apply(init_model)

    def forward(self, x, h_x, a, h_a, feat=None, adj=None,):
        if self.config.use_history_state or self.config.use_history_action:
            hidden = torch.zeros(self.config.gru_num_layers, x.shape[0], self.config.gru_hidden_size).to(
                self.config.device)
            h_input = torch.cat([h_x, h_a], dim=-1) if self.config.use_history_state and self.config.use_history_action \
                else (h_x if self.config.use_history_state else h_a)
            h_input = torch.cat([h_input, self.end_of_token.repeat(h_input.shape[0], 1, 1)], dim=1)
            output, hidden = self.gru(h_input, hidden)
            x = torch.cat([x, output[:, -1, :]], dim=-1) * self.state_mask
        if self.config.use_topology_info:
            output1 = self.gcn(feat, adj)
            x = torch.cat([x, output1], dim=-1)
        x = torch.cat((x, a), dim=1)
        x = F.relu(self.l1(x)) if self.config.active_function == 'relu' else F.tanh(self.l1(x))
        x = F.relu(self.l2(x)) if self.config.active_function == 'relu' else F.tanh(self.l2(x))
        x = self.l3(x)

        return x


class Actor_logProp(nn.Module):
    def __init__(self, settings, config):
        super(Actor_logProp, self).__init__()
        self.settings = settings
        self.config = config
        self.action_std = config.init_action_std if hasattr(config, 'init_action_std') else 0.1
        self.l1_input_dim = config.state_dim
        if self.config.use_history_state or self.config.use_history_action:
            self.l1_input_dim += config.gru_hidden_size
        if self.config.use_topology_info:
            self.l1_input_dim += config.gcn_hidden_size
        self.l1 = nn.Linear(self.l1_input_dim, 512)
        self.l2 = nn.Linear(512, 512)
        # split action header
        self.head_thermal_gen_p = nn.Linear(512, config.more_detail_action_dim['thermal_gen_p'])
        self.head_renewable_gen_p = nn.Linear(512, config.more_detail_action_dim['renewable_gen_p'])
        self.head_adjld_p = nn.Linear(512, config.more_detail_action_dim['adjld_p'])
        self.head_stoenergy_p = nn.Linear(512, config.more_detail_action_dim['stoenergy_p'])
        # use gru to process history state sequence
        if self.config.use_history_state or self.config.use_history_action:
            self.gru_input_size = self.config.state_dim * self.config.use_history_state + \
                                  self.config.action_dim * self.config.use_history_action
            self.gru = nn.GRU(
                input_size=self.gru_input_size,
                hidden_size=self.config.gru_hidden_size,
                num_layers=self.config.gru_num_layers,
                batch_first=True
            )
            self.state_mask = nn.Parameter(
                torch.cat([torch.ones(self.config.state_dim, dtype=torch.float32),
                           torch.zeros(config.gru_hidden_size, dtype=torch.float32)]).unsqueeze(0),
                requires_grad=True,
            )
            self.end_of_token = nn.Parameter(
                torch.zeros(self.gru_input_size, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                requires_grad=True,
            )
        # use gcn to process topology structure
        if self.config.use_topology_info:
            self.gcn = GCN(
                nfeat=self.config.feature_num,
                nhid=self.config.gcn_hidden_size,
                nclass=self.config.gcn_hidden_size,
                dropout=self.config.gcn_dropout,
            )
        # use switch header to control thermal on/off
        if self.config.restrict_thermal_on_off:
            self.head_switch = nn.Linear(512, 2 * config.more_detail_action_dim['thermal_gen_p'])
            self.type_vector = torch.tensor(np.array([0.0, 1.0]), dtype=torch.float32).to(config.device)
        if self.config.init_model:
            self.apply(init_model)

    def forward(self, x, h_x=None, h_a=None, feat=None, adj=None, action_low=None, action_high=None, sample=False):
        if self.config.use_history_state or self.config.use_history_action:
            hidden = torch.zeros(self.config.gru_num_layers, x.shape[0], self.config.gru_hidden_size).to(
                self.config.device)
            h_input = torch.cat([h_x, h_a], dim=-1) if self.config.use_history_state and self.config.use_history_action \
                else (h_x if self.config.use_history_state else h_a)
            h_input = torch.cat([h_input, self.end_of_token.repeat(h_input.shape[0], 1, 1)], dim=1)
            output, hidden = self.gru(h_input, hidden)
            x = torch.cat([x, output[:, -1, :]], dim=-1) * self.state_mask
        if self.config.use_topology_info:
            output1 = self.gcn(feat, adj)
            x = torch.cat([x, output1], dim=-1)
        x = F.relu(self.l1(x)) if self.config.active_function == 'relu' else F.tanh(self.l1(x))
        x = F.relu(self.l2(x)) if self.config.active_function == 'relu' else F.tanh(self.l2(x))
        thermal_gen_p = self.head_thermal_gen_p(x)
        renewable_gen_p = self.head_renewable_gen_p(x)
        adjld_p = self.head_adjld_p(x)
        stoenergy_p = self.head_stoenergy_p(x)
        # switch header output selection for applying action or not
        if self.config.restrict_thermal_on_off:
            thermal_switch = self.head_switch(x).reshape(-1, self.config.more_detail_action_dim['thermal_gen_p'], 2)
            one_hot = nn.functional.gumbel_softmax(thermal_switch, hard=True, dim=-1)
            thermal_switch = (one_hot * self.type_vector).sum(-1)
        # limit action to [0, 1] and reflect yo valid action space
        if self.config.reflect_actionspace and action_low is not None and action_high is not None:
            thermal_gen_p = F.tanh(thermal_gen_p)
            renewable_gen_p = F.tanh(renewable_gen_p)
            adjld_p = F.tanh(adjld_p)
            stoenergy_p = F.tanh(stoenergy_p)

        return self.combine_action(thermal_gen_p, renewable_gen_p,
                                   adjld_p, stoenergy_p,
                                   action_low, action_high, sample,
                                   thermal_switch if self.config.restrict_thermal_on_off
                                   else torch.ones_like(thermal_gen_p).to(self.config.device))

    def combine_action(self, thermal_gen_p, renewable_gen_p, adjld_p, stoenergy_p,
                       action_low, action_high, sample, thermal_switch):
        thermal_gen_p_i, renewable_gen_p_i = 0, 0
        gen_p = torch.zeros([thermal_gen_p.shape[0], self.settings.gen_num]).to(self.config.device)
        gen_switch = torch.ones([thermal_gen_p.shape[0], self.settings.gen_num]).to(self.config.device)
        for i, gen_type in enumerate(self.settings.gen_type):
            if gen_type == 1:
                gen_p[:, i] = thermal_gen_p[:, thermal_gen_p_i]
                gen_switch[:, i] = thermal_switch[:, thermal_gen_p_i]
                thermal_gen_p_i += 1
            elif gen_type == 5:
                gen_p[:, i] = renewable_gen_p[:, renewable_gen_p_i]
                renewable_gen_p_i += 1
        gen_v = torch.zeros_like(gen_p).to(self.config.device)
        # add random noise to sample action
        # only sample at actor stage
        log_prob_output = None
        gen_p_log_prob, gen_v_log_prob, adjld_p_log_prob, stoenergy_p_log_prob = None, None, None, None
        if sample:
            # get gen_p normal distribution for sampling action
            gen_p_dist = Normal(gen_p, self.action_std * self.action_std)
            gen_p = gen_p_dist.sample()
            gen_p_logprob = gen_p_dist.log_prob(gen_p)

            # get adjld_p normal distribution for sampling action
            adjld_p_dist = Normal(adjld_p, self.action_std * self.action_std)
            adjld_p = adjld_p_dist.sample()
            adjld_p_logprob = adjld_p_dist.log_prob(adjld_p)

            # get stoenergy_p normal distribution for sampling action
            stoenergy_p_dist = Normal(stoenergy_p, self.action_std * self.action_std)
            stoenergy_p = stoenergy_p_dist.sample()
            stoenergy_p_logprob = stoenergy_p_dist.log_prob(stoenergy_p)

            # set gen_v_logprob to 0
            gen_v_logprob = torch.zeros_like(gen_p_logprob)

            # cat all logprob togather
            log_prob_output = torch.cat([gen_p_logprob, gen_v_logprob, adjld_p_logprob, stoenergy_p_logprob], dim=-1)
        # cat action by rules
        action_output = torch.cat([gen_p, gen_v, adjld_p, stoenergy_p], dim=-1)
        # record ori action (without reflection, restriction and clamp) for now logprob calculate
        ori_action_output = action_output
        # cat action switch by rules
        action_switch = torch.cat([gen_switch,
                                   torch.ones_like(gen_v).to(self.config.device),
                                   torch.ones_like(adjld_p).to(self.config.device),
                                   torch.ones_like(stoenergy_p).to(self.config.device)], dim=-1)
        # reflect action output to valid action space
        if self.config.reflect_actionspace:
            action_output = (action_high - action_low) * action_output / 2 + (action_high + action_low) / 2
        if self.config.restrict_thermal_on_off:
            action_output *= action_switch

        return action_output, log_prob_output, ori_action_output

    def get_logprob(self, x, a, h_x=None, h_a=None, feat=None, adj=None, action_low=None, action_high=None):
        if self.config.use_history_state or self.config.use_history_action:
            hidden = torch.zeros(self.config.gru_num_layers, x.shape[0], self.config.gru_hidden_size).to(
                self.config.device)
            h_input = torch.cat([h_x, h_a], dim=-1) if self.config.use_history_state and self.config.use_history_action \
                else (h_x if self.config.use_history_state else h_a)
            h_input = torch.cat([h_input, self.end_of_token.repeat(h_input.shape[0], 1, 1)], dim=1)
            output, hidden = self.gru(h_input, hidden)
            x = torch.cat([x, output[:, -1, :]], dim=-1) * self.state_mask
        if self.config.use_topology_info:
            output1 = self.gcn(feat, adj)
            x = torch.cat([x, output1], dim=-1)
        x = F.relu(self.l1(x)) if self.config.active_function == 'relu' else F.tanh(self.l1(x))
        x = F.relu(self.l2(x)) if self.config.active_function == 'relu' else F.tanh(self.l2(x))
        thermal_gen_p = self.head_thermal_gen_p(x)
        renewable_gen_p = self.head_renewable_gen_p(x)
        adjld_p = self.head_adjld_p(x)
        stoenergy_p = self.head_stoenergy_p(x)
        # # switch header output selection for applying action or not
        # if self.config.restrict_thermal_on_off:
        #     thermal_switch = self.head_switch(x).reshape(-1, self.config.more_detail_action_dim['thermal_gen_p'], 2)
        #     one_hot = nn.functional.gumbel_softmax(thermal_switch, hard=True, dim=-1)
        #     thermal_switch = (one_hot * self.type_vector).sum(-1)
        #     thermal_gen_p = thermal_gen_p * thermal_switch
        # limit action to [0, 1] and reflect yo valid action space
        if self.config.reflect_actionspace:
            thermal_gen_p = F.tanh(thermal_gen_p)
            renewable_gen_p = F.tanh(renewable_gen_p)
            adjld_p = F.tanh(adjld_p)
            stoenergy_p = F.tanh(stoenergy_p)
        # get action distribution and calculate logprob
        thermal_gen_p_i, renewable_gen_p_i = 0, 0
        gen_p = torch.zeros([thermal_gen_p.shape[0], self.settings.gen_num]).to(self.config.device)
        for i, gen_type in enumerate(self.settings.gen_type):
            if gen_type == 1:
                gen_p[:, i] = thermal_gen_p[:, thermal_gen_p_i]
                thermal_gen_p_i += 1
            elif gen_type == 5:
                gen_p[:, i] = renewable_gen_p[:, renewable_gen_p_i]
                renewable_gen_p_i += 1
        gen_v = torch.zeros_like(gen_p).to(self.config.device)
        # cat action to get all mean value
        action_mean = torch.cat([gen_p, gen_v, adjld_p, stoenergy_p], dim=-1)
        # get gen_p normal distribution for sampling action
        action_dist = Normal(action_mean, self.action_std * self.action_std)
        action_logprob = action_dist.log_prob(a)
        dist_entropy = action_dist.entropy()

        return action_logprob, dist_entropy


class Critic_Value(nn.Module):
    def __init__(self, settings, config):
        super(Critic_Value, self).__init__()
        self.settings = settings
        self.config = config
        self.l1_input_dim = config.state_dim
        if self.config.use_history_state or self.config.use_history_action:
            self.l1_input_dim += config.gru_hidden_size
        if self.config.use_topology_info:
            self.l1_input_dim += config.gcn_hidden_size
        self.l1 = nn.Linear(self.l1_input_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 1)
        # use gru to process history state sequence
        if self.config.use_history_state or self.config.use_history_action:
            self.gru_input_size = self.config.state_dim * self.config.use_history_state + \
                                  self.config.action_dim * self.config.use_history_action
            self.gru = nn.GRU(
                input_size=self.gru_input_size,
                hidden_size=self.config.gru_hidden_size,
                num_layers=self.config.gru_num_layers,
                batch_first=True
            )
            self.state_mask = nn.Parameter(
                torch.cat([torch.ones(self.config.state_dim, dtype=torch.float32),
                           torch.zeros(config.gru_hidden_size, dtype=torch.float32)]),
                requires_grad=True,
            )
            self.end_of_token = nn.Parameter(
                torch.zeros(self.gru_input_size, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                requires_grad=True,
            )
        # use gcn to process topology structure
        if self.config.use_topology_info:
            self.gcn = GCN(
                nfeat=self.config.feature_num,
                nhid=self.config.gcn_hidden_size,
                nclass=self.config.gcn_hidden_size,
                dropout=self.config.gcn_dropout,
            )
        if self.config.init_model:
            self.apply(init_model)

    def forward(self, x, h_x, h_a, feat=None, adj=None,):
        if self.config.use_history_state or self.config.use_history_action:
            hidden = torch.zeros(self.config.gru_num_layers, x.shape[0], self.config.gru_hidden_size).to(
                self.config.device)
            h_input = torch.cat([h_x, h_a], dim=-1) if self.config.use_history_state and self.config.use_history_action \
                else (h_x if self.config.use_history_state else h_a)
            h_input = torch.cat([h_input, self.end_of_token.repeat(h_input.shape[0], 1, 1)], dim=1)
            output, hidden = self.gru(h_input, hidden)
            x = torch.cat([x, output[:, -1, :]], dim=-1) * self.state_mask
        if self.config.use_topology_info:
            output1 = self.gcn(feat, adj)
            x = torch.cat([x, output1], dim=-1)
        x = F.relu(self.l1(x)) if self.config.active_function == 'relu' else F.tanh(self.l1(x))
        x = F.relu(self.l2(x)) if self.config.active_function == 'relu' else F.tanh(self.l2(x))
        x = self.l3(x)

        return x
