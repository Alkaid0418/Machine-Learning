import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def loss_and_grad(W, b, X, y, reg=0.0):
    n = X.shape[0]
    P = softmax(X @ W + b)
    loss = -np.log(P[np.arange(n), y] + 1e-12).mean()
    loss += 0.5 * reg * np.sum(W * W)

    dlogits = P.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    dW = X.T @ dlogits + reg * W
    db = dlogits.sum(axis=0)
    return loss, dW, db


def make_data(n_per_class=80, D=16, K=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(K, D)) * 2.0
    X = np.vstack([rng.normal(size=(n_per_class, D)) + centers[k] for k in range(K)])
    y = np.repeat(np.arange(K), n_per_class)
    perm = rng.permutation(len(X))
    n = int(len(X) * 0.8)
    tr, te = perm[:n], perm[n:]
    return X[tr], y[tr], X[te], y[te], K


def train(Xtr, ytr, Xte, yte, K, lr=0.1, epochs=50, bs=32, reg=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(Xtr.shape[1], K)) * 0.01
    b = np.zeros(K)
    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            _, dW, db = loss_and_grad(W, b, Xtr[idx], ytr[idx], reg)
            W -= lr * dW
            b -= lr * db
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            tr_loss, _, _ = loss_and_grad(W, b, Xtr, ytr, reg)
            tr_acc = (softmax(Xtr @ W + b).argmax(1) == ytr).mean()
            te_acc = (softmax(Xte @ W + b).argmax(1) == yte).mean()
            print(f"epoch {ep:3d}  loss {tr_loss:.4f}  train_acc {tr_acc:.3f}  test_acc {te_acc:.3f}")
    return W, b


if __name__ == "__main__":
    Xtr, ytr, Xte, yte, K = make_data()
    W, b = train(Xtr, ytr, Xte, yte, K)
    acc = (softmax(Xte @ W + b).argmax(1) == yte).mean()
    print(f"final test acc = {acc:.3f}")
