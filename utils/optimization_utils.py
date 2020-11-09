import torch
import numpy as np
from utils.logging_utils import log

SMALL_CONST = 1e-15


def to_var(p, device):
    return torch.tensor(p, requires_grad=True, device=device)


def kl_loss(probs, unpert_probs):
    unpert_probs = unpert_probs + SMALL_CONST * (unpert_probs <= SMALL_CONST).float().detach()
    correction = SMALL_CONST * (probs <= SMALL_CONST).float().detach()
    corrected_probs = probs + correction.detach()
    kl_loss = (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
    return kl_loss


def perturb_logits(
        unpert_logits,
        stepsize=0.01,
        target_model_wrapper=None,
        num_iterations=3,
        kl_scale=0.01,
        temperature=1.0,
        device="cuda",
        verbose=False,
        logit_mask=0.,
):
    # Generate inital perturbed past
    grad_accumulator = np.zeros(unpert_logits.shape, dtype=np.float32)
    perturbation = to_var(grad_accumulator, device=device)
    optimizer = torch.optim.Adam([perturbation], lr=stepsize)

    # accumulate perturbations for num_iterations
    for i in range(num_iterations):
        optimizer.zero_grad()
        # Compute hidden using perturbed past
        logits = unpert_logits * temperature + perturbation + logit_mask
        probs = torch.softmax(logits / temperature, -1)
        unpert_probs = torch.softmax(unpert_logits, -1)

        loss = torch.scalar_tensor(0.0).to(device)
        loss_list = []

        if target_model_wrapper is not None:
            discrim_loss = target_model_wrapper(probs)
            if verbose and i % 2 == 0:
                log(f"Iteration {i + 1}, pplm_discrim_loss: {discrim_loss.data.cpu().numpy()}")
            loss += discrim_loss
            loss_list.append(discrim_loss)

        if kl_scale > 0.0:
            unpert_probs = unpert_probs + SMALL_CONST * (unpert_probs <= SMALL_CONST).float().to(device).detach()
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            loss += kl_loss

        # compute gradients
        loss.backward()
        optimizer.step()

    # apply the accumulated perturbations to the past
    pert_logits = unpert_logits * temperature + perturbation
    return pert_logits
