"""
-------------------------------------------------
# @Project  :state_space_disaggregation-main
# @File     :covering_pi
# @Date     :2024/12/29 18:29
# @Author   :
-------------------------------------------------
"""
import time
from itertools import chain

import torch
from torch.optim import Adam

from utils.generic_model import GenericModel
from utils.parameter_matrix import ParameterMatrix

# import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class Solver:
    def __init__(
            self,
            env: GenericModel,
            discount: float,
            lr: float = 0.01,
            abstract_rate: float = 0.9,
            max_iter_evaluation: int = int(1e3),
            max_iter_policy_update: int = int(1e4),
            max_time: float = 3e3,
            epsilon_policy_evaluation: float = 1e-3,
    ):
        # Class arguments
        self.env = env
        self.max_time = max_time
        self.pro_sas = []
        for a in range(self.env.action_dim):
            indptr = torch.tensor(self.env.transition_matrix[a].indptr, dtype=torch.int32)  # 行指针
            indices = torch.tensor(self.env.transition_matrix[a].indices, dtype=torch.int32)  # 列索引
            values = torch.tensor(self.env.transition_matrix[a].data)  # 非零值

            # 转换为 PyTorch 稀疏 CSR 张量
            sparse_tensor = torch.sparse_csr_tensor(
                crow_indices=indptr,
                col_indices=indices,
                values=values,
                size=self.env.transition_matrix[a].shape,
                dtype=torch.float32,
                device=device  # 可选 "cuda" 如果需要在 GPU 上
            )
            self.pro_sas.append(sparse_tensor)
        self.reward_matrix = torch.as_tensor(self.env.reward_matrix, dtype=torch.float32).to(device)

        self.lr = lr
        self.abstract_state_num = int(self.env.state_dim * 0.8)

        self.discount = discount
        self.max_iter_evaluation = max_iter_evaluation
        self.max_iter_policy_update = max_iter_policy_update
        self.name = "pi_ebhpg_08"

        self.epsilon_policy_evaluation = epsilon_policy_evaluation

        self.policy_theta = ParameterMatrix(self.env.state_dim, self.env.action_dim).to(device)

        self.nu_omega = ParameterMatrix(self.abstract_state_num, self.env.state_dim).to(device)

        self.mu_omega = ParameterMatrix(self.env.state_dim, self.abstract_state_num).to(device)
        self.optimizer = Adam(chain(
            self.policy_theta.parameters(),
            self.nu_omega.parameters(),
            self.mu_omega.parameters()
        ), lr=0.01)
        self.value = torch.zeros((self.abstract_state_num, self.env.action_dim), dtype=torch.float32).to(device)

    def run(self):
        start_time = time.time()

        print_dict = {}
        value_list = []
        i_episode = 0
        gap_time = 0
        while True:
            with torch.no_grad():
                self.value, p, r_sa, _ = self._policy_evaluation(
                    self.policy_theta.forward().detach(), self.epsilon_policy_evaluation, self.max_iter_evaluation
                )
                tem_value = self._policy_evaluation_stand(self.policy_theta.forward().detach(),
                                                          self.max_iter_evaluation)
                tem_print = torch.mean(tem_value).item()
                if i_episode == 0:
                    end_time = time.time()
                    print_dict.update({'{}'.format(i_episode): [tem_print, end_time - start_time]})
                    print("Episode: {0:<5} Time :{2:5.2f} | Value {1:5.2f}".format(0, tem_print, end_time - start_time))
                q_value = self.bellman_w(self.value)
            # update policy
            for _ in range(10):
                # ori
                # loss_kl = torch.sum((p.detach() - self.mu_omega.forward() @ self.nu_omega.forward()) ** 2)
                # policy_theta_loss = - (self.policy_theta.forward() * q_value)
                # nu_omega_loss = - (self.nu_omega.forward() * q_value.sum(-1, keepdim=True).T)
                # p_next_s = (self.nu_omega.forward() @ p).detach()
                # mu_omega_loss = - (p_next_s @ self.mu_omega.forward() * self.value.reshape(1, -1))

                value_ture = torch.inverse((torch.eye(self.env.state_dim).to(device) - self.discount * p)) @ r_sa
                u_p = self.nu_omega.forward() @ p @ self.mu_omega.forward()
                nu_m = self.nu_omega.forward()
                loss_kl = torch.linalg.norm((nu_m @ p.detach() - u_p @ nu_m) @ value_ture.detach())
                policy_theta_loss = - (self.policy_theta.forward() * q_value)
                v = torch.linalg.norm(self.value).item()
                loss = policy_theta_loss.mean() + v * loss_kl

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # lower bounded
            time_1 = time.time()
            nu_m = nu_m.detach()
            v_u = torch.mean(self.value)
            lower_bounded = torch.linalg.norm((nu_m @ p.detach() - u_p.detach() @ nu_m) @ value_ture.detach()) / (
                        1 - self.discount)
            lower_bounded = v_u.item() - lower_bounded
            time_2 = time.time()
            gap_time += time_2 - time_1

            # self.nu = self.nu_omega / torch.sum(self.nu_omega, dim=1, keepdims=True)
            value_list.append(tem_print)
            if i_episode % 4 == 0:
                end_time = time.time()
                print("Episode: {0:<5} Time :{2:5.2f} | Value {1:5.2f},"
                      " Lower Bounded {3:5.2f} hat_V {4:5.2f}".format(i_episode,
                                                       tem_print, end_time - start_time - gap_time,
                                                       lower_bounded, v_u.item()))
                print_dict.update({'{}'.format(i_episode): [tem_print, end_time - start_time - gap_time,
                                                            lower_bounded, v_u.item()]})
                value_list = []
                gap_time = 0
            i_episode += 1
            end_time = time.time() - start_time
            if i_episode >= self.max_time:
                break
        return print_dict

    def _policy_evaluation(
            self,
            policy: torch.Tensor,
            epsilon_policy_evaluation: float,
            max_iteration_evaluation: int,
    ):
        with torch.no_grad():
            nu_dagger = self.mu_omega.forward().detach()

            eval_iter = 0
            transition_policy_ori, reward_policy_ori = self._compute_transition_reward_pi(policy)
            nu = self.nu_omega.forward().detach()
            transition_policy = nu @ transition_policy_ori @ nu_dagger

            reward_policy = nu @ reward_policy_ori
            value = torch.zeros((self.abstract_state_num,), dtype=torch.float32).to(device)
            # new_value = torch.inverse(torch.eye(self.abstract_state_num, dtype=torch.float32).to(device)
            #                           - self.discount * transition_policy) @ reward_policy

            for i in range(max_iteration_evaluation):
                eval_iter += 1
                new_value = reward_policy + self.discount * transition_policy @ value
                variation = torch.absolute(new_value - value).max()
                if ((variation < ((1 - self.discount) / self.discount) * epsilon_policy_evaluation) or
                        eval_iter >= max_iteration_evaluation):
                    return new_value, transition_policy_ori, reward_policy_ori, transition_policy
                else:
                    value = new_value

        return new_value, transition_policy_ori, reward_policy_ori, transition_policy

    def _compute_transition_reward_pi(self, policy):
        policy_t = policy.unsqueeze(-1)
        matrix_all = torch.zeros((self.env.state_dim, self.env.state_dim), dtype=torch.float32).to(device)
        for i in range(len(self.pro_sas)):
            matrix_all += policy_t[:, i] * self.pro_sas[i].to_dense()

        reward_policy = (policy * self.reward_matrix).sum(1)
        return matrix_all, reward_policy

    def bellman_w(self, value):
        q_value = torch.zeros((self.env.state_dim, self.env.action_dim), dtype=torch.float32).to(device)
        p_nv_dagger = self.mu_omega.forward().detach()
        for aa in range(self.env.action_dim):
            p_sas = self.pro_sas[aa].to_dense()
            next_value_i = p_sas @ p_nv_dagger @ value
            q_value[:, aa] = (
                    self.reward_matrix[:, aa]
                    + self.discount * next_value_i
            )

        return q_value

    def _policy_evaluation_stand(
            self,
            policy: torch.Tensor,
            max_iteration_evaluation: int,
    ) -> torch.Tensor:
        eval_iter = 0
        transition_policy, reward_policy = self._compute_transition_reward_pi_stand(policy)
        value = torch.zeros((self.env.state_dim,), dtype=torch.float32).to(device)
        for i in range(max_iteration_evaluation):
            eval_iter += 1
            new_value = reward_policy + self.discount * transition_policy @ value
            value = new_value
        return new_value

    def _compute_transition_reward_pi_stand(self, policy):
        policy_t = policy.unsqueeze(-1)
        matrix_all = torch.zeros((self.env.state_dim, self.env.state_dim), dtype=torch.float32).to(device)
        for i in range(len(self.pro_sas)):
            matrix_all += policy_t[:, i] * self.pro_sas[i].to_dense()

        reward_policy = (policy * self.reward_matrix).sum(1)
        return matrix_all, reward_policy
