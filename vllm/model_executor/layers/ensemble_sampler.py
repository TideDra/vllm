from typing import Optional, Dict, Tuple, List

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.rejection_sampler import (
    RejectionSampler, _multinomial)

logger = init_logger(__name__)

class EnsembleSampler(RejectionSampler):
    
    def __init__(self,
                 strict_mode: bool = False,
                 use_flashinfer: Optional[bool] = None):
        if use_flashinfer:
            logger.info("Flashinfer is not compatiable with ensemble sampler.")
        super().__init__(strict_mode=strict_mode, use_flashinfer=False)
        
    
    def forward(
        self,
        target_with_bonus_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        acceptance_ensemble_lambda: Optional[List[float]] = None,
        distribution_ensemble_lambda: Optional[List[float]] = None,
        seeded_seqs: Optional[Dict[int, torch.Generator]] = None,
    ) -> torch.Tensor:
        # Only perform shape/dtype/device checking in strict mode, as it adds
        # overhead.
        if self._strict_mode:
            self._raise_if_incorrect_input(target_with_bonus_probs,
                                           draft_token_ids, bonus_token_ids,
                                           draft_probs)

        batch_size, k, vocab_size = draft_probs.shape

        # batch_size = 0 when all requests in the batch are
        # non_spec requests. In this case, output_token_ids is
        # just an empty tensor.
        if batch_size == 0:
            return torch.empty(0, k + 1, device=draft_probs.device, dtype=int)

        target_probs = target_with_bonus_probs[:, :-1]
        if distribution_ensemble_lambda is not None:
            lambda_d = torch.tensor(distribution_ensemble_lambda,
                                    dtype=draft_probs.dtype,
                                    device=draft_probs.device).view(-1, 1, 1)
            target_probs = lambda_d * target_probs + (1 - lambda_d) * draft_probs

        accepted = self._get_accepted(target_probs,
                                      draft_probs,
                                      draft_token_ids,
                                      acceptance_ensemble_lambda,
                                      seeded_seqs)
        
        recovered_probs = self._get_recovered_probs(
            target_probs, draft_probs).reshape(batch_size * k, vocab_size)

        # NOTE: the recovered_probs are overwritten by this method.
        recovered_token_ids = _multinomial(
            recovered_probs,
            num_samples=1,
            k=k,
            seeded_seqs=seeded_seqs or {},
        ).reshape(batch_size, k)

        
        output_token_ids = self._create_output(
            accepted,
            recovered_token_ids,
            draft_token_ids,
            -torch.ones_like(bonus_token_ids), # no bonus token
        )

        return output_token_ids
    
    def _get_accepted(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        acceptance_ensemble_lambda: Optional[List[float]],
        seeded_seqs: Optional[Dict[int, torch.Generator]],
    ) -> torch.Tensor:
        r"""Create bool matrix over the proposed draft tokens. If
        True, then a token can be accepted, else it should be
        rejected.

        Given :math:`q(\hat{x}_{n+1}|x_1, \dots, x_n)`, the probability of
        :math:`\hat{x}_{n+1}` given context :math:`x_1, \dots, x_n` according
        to the target model, and :math:`p(\hat{x}_{n+1}|x_1, \dots, x_n)`, the
        same conditional probability according to the draft model, the token
        is accepted with probability:

        .. math::
            \min\left(1, \frac{q(\hat{x}_{n+1}|x_1, \dots, x_n)}
                           {\lambda_a p(\hat{x}_{n+1}|x_1, \dots, x_n)}\right)

        This implementation does not apply causality. When using the output,
        if a token is rejected, subsequent tokens should not be used.

        Returns a bool tensor of shape [batch_size, k] specifying which tokens
        are accepted.
        """
        batch_size, k, _ = draft_probs.shape

        batch_indices = torch.arange(batch_size,
                                     device=target_probs.device)[:, None]
        probs_indicies = torch.arange(k, device=target_probs.device)

        # shape [batch_size, k]
        selected_draft_probs = draft_probs[batch_indices, probs_indicies,
                                           draft_token_ids]

        # shape [batch_size, k]
        selected_target_probs = target_probs[batch_indices, probs_indicies,
                                             draft_token_ids]
        
        if acceptance_ensemble_lambda is not None:
            lambda_a = torch.tensor(acceptance_ensemble_lambda,
                                    dtype=draft_probs.dtype,
                                    device=draft_probs.device).view(-1, 1)
            lambda_a.clamp_min_(1e-4)
            accept_ratio = selected_target_probs / (selected_draft_probs * lambda_a)
        else:
            accept_ratio = selected_target_probs / selected_draft_probs

        uniform_rand = self._create_uniform_samples(seeded_seqs, batch_size,
                                                    k - 1, target_probs.device)
        
        accepted = uniform_rand < accept_ratio

        return accepted