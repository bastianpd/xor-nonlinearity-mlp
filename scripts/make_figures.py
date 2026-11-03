"""Regenerates the presentation figures at high resolution from the notebook.

It executes the *computational* cells of notebooks/xor_WhyNonLinearity_MLP.ipynb
(same seeds, same data, same training) and re-draws the plots with larger fonts
and one message per figure so they stay legible inside docs/index.html.
Nothing in the notebook is modified.

Usage:  python scripts/make_figures.py
Needs:  numpy, pandas, matplotlib
"""
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "xor_WhyNonLinearity_MLP.ipynb"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

cells = json.loads(NB.read_text(encoding="utf8"))["cells"]
ns = {"display": lambda d: print(d.round(4) if hasattr(d, "round") else d)}
# computational cells only (imports, data, maths, training, tables)
for i in (3, 5, 7, 9, 12, 14, 16, 18, 20, 22):
    exec("".join(cells[i]["source"]), ns)
plt.close("all")

g = ns  # shorthand
X_train, y_train, X_test, y_test = g["X_train"], g["y_train"], g["X_test"], g["y_test"]
H_l0, H_l1, hist_lr = g["H_l0"], g["H_l1"], g["hist_lr"]
P_l0, P_l1, w_lr, b_lr = g["P_l0"], g["P_l1"], g["w_lr"], g["b_lr"]
forward, sigmoid, EPOCHS = g["forward"], g["sigmoid"], g["EPOCHS"]

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 200, "font.size": 13,
                     "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 12,
                     "axes.grid": True, "grid.alpha": .25})

gx = np.linspace(-2.6, 2.6, 400)
GX, GY = np.meshgrid(gx, gx)
GRID = np.c_[GX.ravel(), GY.ravel()]


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def points(ax, s_tr=22, s_te=48):
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train[:, 0], cmap="coolwarm", edgecolor="k", s=s_tr, linewidth=.35, label="train")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test[:, 0], cmap="coolwarm", edgecolor="k", s=s_te, marker="X", linewidth=.5, label="test")


# 1 ── dataset ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 6))
for X, y, mk, nm in [(X_train, y_train, "o", "train"), (X_test, y_test, "X", "test")]:
    for c, col in [(0, "tab:blue"), (1, "tab:red")]:
        m = y[:, 0] == c
        ax.scatter(X[m, 0], X[m, 1], c=col, marker=mk, s=26 if mk == "o" else 55, edgecolor="k", linewidth=.4, label=f"class {c} ({nm})")
ax.set_title("XOR dataset: 4 Gaussian clusters, σ = 0.35\n300 train (●) / 100 test (✖)")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal"); ax.legend(loc="upper center", ncol=2, fontsize=10, framealpha=.9)
ax.set_ylim(-2.6, 2.9)
save(fig, "dataset.png")

# 2 ── loss curves (one message per figure, legible at half page width) ───
ep = np.arange(1, EPOCHS + 1)
fig, ax = plt.subplots(figsize=(7.4, 5))
ax.plot(np.arange(1, len(hist_lr["loss_tr"]) + 1), hist_lr["loss_tr"], label="Logistic regression")
ax.plot(ep, H_l0["loss_bce_tr"], label="MLP λ = 0")
ax.plot(ep, H_l1["loss_bce_tr"], "--", label="MLP λ = 0.005")
ax.set(title="BCE on train (without regularizer)", xscale="log", xlabel="epochs", ylabel="BCE"); ax.legend()
fig.tight_layout(); save(fig, "loss_bce.png")

fig, ax = plt.subplots(figsize=(7.4, 5))
ax.plot(ep, H_l0["loss_total_tr"], label="$L_{total}$ (λ = 0)")
ax.plot(ep, H_l1["loss_total_tr"], "--", label="$L_{total}$ (λ = 0.005)")
ax.plot(ep, H_l1["R_tr"], ":", lw=2.2, label="R term (λ = 0.005)")
ax.set(title="Total objective and L1 term", xscale="log", xlabel="epochs"); ax.legend()
fig.tight_layout(); save(fig, "loss_total.png")

# 3 ── sparsity during training ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.4, 5))
ax.plot(ep, H_l0["frac_zero_tr"], label="λ = 0")
ax.plot(ep, H_l1["frac_zero_tr"], "--", label="λ = 0.005")
ax.set(title="Fraction of hidden activations = 0 (train)", xscale="log", xlabel="epochs", ylabel="fraction")
ax.legend(); fig.tight_layout(); save(fig, "sparsity.png")


# 4 ── probability maps (one panel each) ──────────────────────────────────
def pmap(ax, probs, title):
    cs = ax.contourf(GX, GY, probs.reshape(GX.shape), levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=.85)
    ax.contour(GX, GY, probs.reshape(GX.shape), levels=[.5], colors="k", linewidths=2.4)
    points(ax, 16, 36)
    ax.set_title(title); ax.set_aspect("equal"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.grid(False)
    return cs

p_lr = sigmoid(GRID @ w_lr + b_lr); p_l0 = forward(GRID, P_l0)[0]; p_l1 = forward(GRID, P_l1)[0]
for probs, title, name in [
        (p_lr, f"Logistic regression\ntest accuracy = {hist_lr['acc_te'][-1]:.2f}", "boundary_logreg.png"),
        (p_l0, f"MLP 2→4→1, λ = 0\ntest accuracy = {H_l0['acc_te'][-1]:.2f}", "boundary_lambda0.png"),
        (p_l1, f"MLP 2→4→1, λ = 0.005\ntest accuracy = {H_l1['acc_te'][-1]:.2f}", "boundary_lambda005.png")]:
    fig, ax = plt.subplots(figsize=(6.8, 5.9))
    cs = pmap(ax, probs, title)
    fig.colorbar(cs, ax=ax, shrink=.85, pad=.02, label="P(class 1)"); save(fig, name)


# 5 ── hidden neurons: 2×2 grid per λ ─────────────────────────────────────
def hidden(P, title, name):
    _, c = forward(GRID, P)
    fig, axs = plt.subplots(2, 2, figsize=(11, 10.2))
    for j, ax in enumerate(axs.ravel()):
        im = ax.contourf(GX, GY, c["A1"][:, j].reshape(GX.shape), levels=20, cmap="viridis")
        ax.contour(GX, GY, c["Z1"][:, j].reshape(GX.shape), levels=[0], colors="w", linewidths=3)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train[:, 0], cmap="coolwarm", edgecolor="k", s=14, linewidth=.3)
        w = P["W1"][:, j]
        ax.set_title(f"$A_{{1,{j+1}}}$:  w = ({w[0]:.2f}, {w[1]:.2f}),  b = {P['b1'][0, j]:.2f}", fontsize=13)
        ax.set_aspect("equal"); ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gx[0], gx[-1]); ax.grid(False)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle(title, fontsize=15); fig.tight_layout(); save(fig, name)

hidden(P_l0, "Hidden neurons, λ = 0  (white line: $Z_{1,j}=0$)", "hidden_lines_lambda0.png")
hidden(P_l1, "Hidden neurons, λ = 0.005  (white line: $Z_{1,j}=0$)", "hidden_lines_lambda005.png")


# 6 ── linear regions + final boundary ────────────────────────────────────
def regions(P, title, name):
    _, c = forward(GRID, P)
    pattern = ((c["Z1"] > 0) * (2 ** np.arange(4))).sum(axis=1)
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.imshow(pattern.reshape(GX.shape), origin="lower", extent=[gx[0], gx[-1], gx[0], gx[-1]], cmap="tab20", alpha=.45)
    for j in range(4):
        ax.contour(GX, GY, c["Z1"][:, j].reshape(GX.shape), levels=[0], colors=f"C{j}", linewidths=2, linestyles="--")
        ax.plot([], [], color=f"C{j}", ls="--", label=f"$Z_{{1,{j+1}}}=0$")
    ax.contour(GX, GY, c["Z2"].reshape(GX.shape), levels=[0], colors="k", linewidths=3.2)
    ax.plot([], [], color="k", lw=3, label="boundary $Z_2=0$")
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train[:, 0], cmap="coolwarm", edgecolor="k", s=16, linewidth=.3)
    ax.set_title(f"{title}\n{len(np.unique(pattern))} activation patterns in the window")
    ax.set_aspect("equal"); ax.grid(False); ax.legend(fontsize=10, loc="upper right", framealpha=.9)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); save(fig, name)

regions(P_l0, "Linear regions and boundary, λ = 0", "regions_lambda0.png")
regions(P_l1, "Linear regions and boundary, λ = 0.005", "regions_lambda005.png")

# 7 ── gradient check on the trained λ=0 model (same function as the notebook) ──
print("\n[gradient check, trained MLP λ=0]")
g["gradient_check"](P_l0, X_train, y_train, 0.0, label="trained model")
