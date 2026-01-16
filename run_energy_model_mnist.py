# imports <<<
import argparse
import os
import json
import logging
import sys
import torch
import matplotlib.pyplot as plt

from lib import config, data, energy, train, utils
# >>>

# load default config <<<
def load_default_config():
    default_config = "./etc/hopfield_train_output.json"
    # default_config = "./etc/hopfield_train_input_output.json"
    # default_config = "./etc/hopfield_reconstruct.json"
    with open(default_config) as f:
        return json.load(f)
# >>>

# parse shell arguments <<<
def parse_shell_args(args):
    parser = argparse.ArgumentParser(
        description="Train a Hopfield EBM on MNIST using Equilibrium Propagation."
    )

    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--c_energy", choices=["cross_entropy", "squared_error"], default=argparse.SUPPRESS)
    parser.add_argument("--dimensions", type=int, nargs="+", default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--learning_rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--nonlinearity", choices=["leaky_relu", "relu", "sigmoid", "tanh"], default=argparse.SUPPRESS)
    parser.add_argument("--optimizer", choices=["adam", "adagrad", "sgd"], default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)

    parser.add_argument("--train_subset", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--save_weights", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--load_weights", type=str, default=argparse.SUPPRESS)

    parser.add_argument("--mode", choices=["train", "gen"], default=argparse.SUPPRESS)

    parser.add_argument("--gen_input_mode", choices=["corrupt_mnist", "random"], default=argparse.SUPPRESS)
    parser.add_argument("--gen_sample_index", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--gen_corrupt_frac", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gen_corrupt_mode", choices=["zero", "noise"], default=argparse.SUPPRESS)
    parser.add_argument("--gen_output_class", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--gen_output_strength", type=float, default=argparse.SUPPRESS)

    parser.add_argument("--input_train", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--input_train_frac", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--input_mse_weight", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--input_corrupt_mode", choices=["zero", "noise"], default=argparse.SUPPRESS)

    return vars(parser.parse_args(args))

# >>>

# generate one mnist sample <<<
def get_one_mnist_sample(mnist_train_loader, index, device, batch_size):
    ds = mnist_train_loader.dataset
    n = len(ds)

    if index is None or index < 0:
        index = int(torch.randint(low=0, high=n, size=(1,)).item())

    x, y = ds[index]
    x = x.view(1, -1).to(device)
    y = y.view(1, -1).to(device)

    if batch_size > 1:
        x = x.repeat(batch_size, 1)
        y = y.repeat(batch_size, 1)

    return x, y, index
# >>>

# corrupt mnist sample <<<
def corrupt_input(x, corrupt_frac, mode="zero"):
    if corrupt_frac < 0 or corrupt_frac > 1:
        raise ValueError(f"gen_corrupt_frac must be in [0,1], got {corrupt_frac}")

    corrupt_mask = (torch.rand_like(x) < corrupt_frac)
    x_cor = x.clone()

    if mode == "zero":
        x_cor[corrupt_mask] = 0.0
    elif mode == "noise":
        x_cor[corrupt_mask] = torch.randn_like(x_cor[corrupt_mask])
    else:
        raise ValueError(f'gen_corrupt_mode "{mode}" not defined (use "zero" or "noise").')

    return x_cor, corrupt_mask
# >>>

# make output logits <<<
def make_output_logits(batch_size, n_classes, class_id, strength, device):
    logits = torch.zeros((batch_size, n_classes), device=device)
    logits[:, int(class_id)] = float(strength)
    return logits
# >>>

# convert state to image <<<
def u0_to_mnist_image(u0_batch, batch_index=0):
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    x = u0_batch[batch_index].detach().cpu().view(28, 28)
    x = x * MNIST_STD + MNIST_MEAN
    x = torch.clamp(x, 0.0, 1.0)
    return x
# >>>

# plot side-by-side figures <<<
def plot_side_by_side(x0, x1):
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(x0, cmap="gray")
    plt.axis("off")
    plt.title("Initial u[0]")

    plt.subplot(1, 2, 2)
    plt.imshow(x1, cmap="gray")
    plt.axis("off")
    plt.title("Final u[0]")

    plt.tight_layout()
    plt.show()
    return
# >>>

# run energy model mnist <<<
def run_energy_model_mnist(cfg):
    if cfg.get("seed", None) is not None:
        torch.manual_seed(cfg["seed"])

    # initialize stuff <<<
    c_energy = utils.create_cost(cfg["c_energy"], cfg["beta"])
    phi = utils.create_activations(cfg["nonlinearity"], len(cfg["dimensions"]))
    model = energy.HopfieldEBM(cfg["dimensions"], c_energy, cfg["batch_size"], phi).to(config.device)
    w_optimizer = utils.create_optimizer(model, cfg["optimizer"], lr=cfg["learning_rate"])
    # >>>

    # load stored weights <<<
    load_path = cfg.get("load_weights", "")
    if load_path:
        state = torch.load(load_path, map_location=config.device)
        model.load_state_dict(state, strict=True)
        logging.info(f"Loaded weights from: {load_path}")
    # >>>

    # create mnist dataset loaders <<<
    mnist_train, mnist_test = data.create_mnist_loaders(
        cfg["batch_size"],
        train_subset=cfg.get("train_subset", 1.0),
        seed=cfg.get("seed", None),
    )
    # >>>

    mode = cfg.get("mode", "train")

    if mode == "gen":
        states = run_generation(model, mnist_train, cfg)
        x0 = u0_to_mnist_image(states["u_init"][0], batch_index=0)
        x1 = u0_to_mnist_image(states["u"][0], batch_index=0)
        plot_side_by_side(x0, x1)
        return

    logging.info("Start training with parametrization:\n{}".format(
        json.dumps(cfg, indent=4, sort_keys=True)))

    for epoch in range(1, cfg["epochs"] + 1):
        train.train(model, mnist_train, cfg, w_optimizer, cfg.get("fast_ff_init", False))
        test_acc, test_energy = train.test(model, mnist_test, cfg["dynamics"], cfg.get("fast_ff_init", False))
        logging.info("epoch: {} \t test_acc: {:.4f} \t mean_E: {:.4f}".format(epoch, test_acc, test_energy))

    save_path = cfg.get("save_weights", "")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logging.info(f"Saved weights to: {save_path}")
# >>>

# use model for generation <<<
def run_generation(model, mnist_train_loader, cfg):
    model.reset_state()
    model.set_C_target(None)

    input_mode = cfg.get("gen_input_mode", "corrupt_mnist")

    idx = None
    y_true = None
    x_clean = None
    corrupt_mask = None

    if input_mode == "corrupt_mnist":
        x_clean, y_true, idx = get_one_mnist_sample(
            mnist_train_loader,
            cfg.get("gen_sample_index", -1),
            config.device,
            model.batch_size,
        )

        x_cor, corrupt_mask = corrupt_input(
            x_clean,
            corrupt_frac=float(cfg.get("gen_corrupt_frac", 0.0)),
            mode=str(cfg.get("gen_corrupt_mode", "zero")),
        )

        model.u[0] = x_cor.detach()
        model.release_layer(0)
        model.update_energy()

        u0_init = model.u[0].detach().cpu().clone()

    elif input_mode == "random":
        model.release_layer(0)
        model.update_energy()
        u0_init = model.u[0].detach().cpu().clone()

    else:
        raise ValueError(f'gen_input_mode "{input_mode}" not defined.')

    out_class = int(cfg.get("gen_output_class", -1))
    if out_class >= 0:
        logits = make_output_logits(
            batch_size=model.batch_size,
            n_classes=model.dimensions[-1],
            class_id=out_class,
            strength=float(cfg.get("gen_output_strength", 1.0)),
            device=config.device,
        )
        model.clamp_layer(model.n_layers - 1, logits)

    if input_mode == "corrupt_mnist":
        u_relax_input_inpaint(model, cfg["dynamics"], corrupt_mask=corrupt_mask)
    else:
        model.u_relax(**cfg["dynamics"])

    return {
        "u_init": [u0_init],
        "u": [u_i.detach().cpu() for u_i in model.u],
        "sample_index": idx,
        "true_label": None if y_true is None else int(torch.argmax(y_true[0]).item()),
        "gen_output_class": out_class,
        "gen_input_mode": input_mode,
    }

def u_relax_input_inpaint(model, dynamics, corrupt_mask):
    model.release_layer(0)

    dt = dynamics["dt"]
    tau = dynamics["tau"]
    tol = dynamics["tol"]
    n_relax = dynamics["n_relax"]

    for _ in range(n_relax):
        du_norm = model.u_step(dt, tau, u0_free_mask=corrupt_mask)
        if du_norm < tol:
            break
# >>>

if __name__ == "__main__":
    user_config = parse_shell_args(sys.argv[1:])
    cfg = load_default_config()
    cfg.update(user_config)

    config.setup_logging("hopfield_" + cfg["c_energy"] + "_" + cfg["dataset"], dir=cfg.get("log_dir", ""))

    run_energy_model_mnist(cfg)

