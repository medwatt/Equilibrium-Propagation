# MIT License
# Copyright (c) 2020 Simon Schug, João Sacramento

import logging
import torch

from lib import config


def predict_batch(model, x_batch, dynamics, fast_init):
    """
    Standard prediction: clamp full input, no input reconstruction term.
    """
    model.reset_state()

    # clamp input and remove supervised nudging
    model.clamp_layer(0, x_batch.view(-1, model.dimensions[0]))
    model.set_C_target(None)

    # IMPORTANT: prediction shouldn't include input recon term
    if hasattr(model, "clear_input_target"):
        model.clear_input_target()

    if fast_init:
        model.fast_init()
    else:
        model.u_relax(**dynamics)

    return torch.nn.functional.softmax(model.u[-1].detach(), dim=1)


def test(model, test_loader, dynamics, fast_init):
    test_E, correct, total = 0.0, 0.0, 0.0

    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)

        output = predict_batch(model, x_batch, dynamics, fast_init)
        prediction = torch.argmax(output, 1)

        with torch.no_grad():
            correct += float(torch.sum(prediction == y_batch.argmax(dim=1)))
            test_E += float(torch.sum(model.E))
            total += x_batch.size(0)

    return correct / total, test_E / total


def _relax_with_u0_mask(model, dynamics, u0_free_mask):
    """
    Relaxation where only masked pixels of u0 are allowed to change.
    Fixed pixels never change (masked inside u_step).
    """
    dt = dynamics["dt"]
    tau = dynamics["tau"]
    tol = dynamics["tol"]
    n_relax = dynamics["n_relax"]

    for _ in range(n_relax):
        du_norm = model.u_step(dt, tau, u0_free_mask=u0_free_mask)
        if du_norm < tol:
            break


def train(model, train_loader, cfg, w_optimizer, fast_init):
    """
    Equilibrium propagation training.

    Inpainting mode (input_train=True and input_train_frac>0):
      - corrupt input pixels at fraction input_train_frac
      - set u0 to corrupted input (x_cor)
      - keep uncorrupted pixels fixed forever (only corrupted pixels update)
      - add input MSE only on corrupted pixels (toward clean x)
      - supervised output cost stays the same
    """
    dynamics = cfg["dynamics"]

    input_train = bool(cfg.get("input_train", False))
    corrupt_frac = float(cfg.get("input_train_frac", 0.0))
    input_mse_weight = float(cfg.get("input_mse_weight", 0.0))

    # corruption mode for training (reuse your gen_corrupt_mode or default to "zero")
    corrupt_mode = str(cfg.get("input_corrupt_mode", "zero"))  # "zero" or "noise"

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)

        model.reset_state()

        # clean reference input
        x_clean = x_batch.view(-1, model.dimensions[0])

        # -------------------------
        # INPAINT TRAINING BRANCH
        # -------------------------
        if input_train and corrupt_frac > 0.0:
            # mask True where pixel is corrupted (free)
            corrupt_mask = (torch.rand_like(x_clean) < corrupt_frac)

            # build corrupted input
            x_cor = x_clean.clone()
            if corrupt_mode == "zero":
                x_cor[corrupt_mask] = 0.0
            elif corrupt_mode == "noise":
                x_cor[corrupt_mask] = torch.randn_like(x_cor[corrupt_mask])
            else:
                raise ValueError(f'input_corrupt_mode "{corrupt_mode}" not defined (use "zero" or "noise").')

            # initialize u0 to corrupted input (NOT clamped)
            model.u[0] = x_cor.detach()
            model.release_layer(0)

            # input recon: penalize ONLY corrupted pixels toward clean x
            if hasattr(model, "set_input_target") and input_mse_weight > 0.0:
                model.set_input_target(x_clean, x0_mask=corrupt_mask, weight=input_mse_weight)

            # Free phase (no supervised nudging)
            model.set_C_target(None)
            _relax_with_u0_mask(model, dynamics, u0_free_mask=corrupt_mask)
            free_grads = model.w_get_gradients()

            # Nudged phase (supervised)
            model.set_C_target(y_batch)
            _relax_with_u0_mask(model, dynamics, u0_free_mask=corrupt_mask)
            nudged_grads = model.w_get_gradients()

        # -------------------------
        # STANDARD TRAINING BRANCH
        # -------------------------
        else:
            # clamp full input
            model.clamp_layer(0, x_clean)

            # ensure no input recon term is active
            if hasattr(model, "clear_input_target"):
                model.clear_input_target()

            # Free phase
            if fast_init:
                model.fast_init()
                free_grads = [torch.zeros_like(p) for p in model.parameters()]
            else:
                model.set_C_target(None)
                _ = model.u_relax(**dynamics)
                free_grads = model.w_get_gradients()

            # Nudged phase
            model.set_C_target(y_batch)
            _ = model.u_relax(**dynamics)
            nudged_grads = model.w_get_gradients()

        # weight update
        model.w_optimize(free_grads, nudged_grads, w_optimizer)

        # Logging
        if batch_idx % (len(train_loader) // 10) == 0:
            output = predict_batch(model, x_batch, dynamics, fast_init)
            prediction = torch.argmax(output, 1)
            batch_acc = float(torch.sum(prediction == y_batch.argmax(dim=1))) / x_batch.size(0)
            logging.info('{:.0f}%:\tE: {:.2f}\tbatch_acc {:.4f}'.format(
                100. * batch_idx / len(train_loader), torch.mean(model.E), batch_acc))

