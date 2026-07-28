"""Treino com data augmentation -- testa se ela fecha o gap entre sessoes.

    .venv\\Scripts\\python.exe training\\train_aug.py

Parte dos LANDMARKS CRUS (world), nao das features prontas -- porque a
augmentation (rotacao/espelho) age sobre a nuvem 3D antes da normalizacao.
Aplica augmentation SO no treino de cada dobra LeaveOneGroupOut; o teste fica
cru. Compara com o baseline sem augmentation (LogReg: media 95.9%, dificil 88.4%,
T/F entre sessoes 80%, R/U 70%).
"""

from __future__ import annotations

import gc
import os

# BLAS de uma thread so: com muitas amostras (pos-augmentation) o OpenBLAS
# multi-thread no Windows estourou memoria. Precisa vir antes de carregar numpy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from sklearn.linear_model import SGDClassifier  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from libras.data.augment import aumentar_treino  # noqa: E402
from libras.data.dataset import carregar_dataset  # noqa: E402
from libras.vision.features import normalizar_mao  # noqa: E402


def _features(world: np.ndarray) -> np.ndarray:
    return np.array([normalizar_mao(w) for w in world], dtype=np.float32)


def _pipe():
    # SGDClassifier(log_loss) = MESMO modelo linear-logistico da baseline, mas
    # treinado por minilotes (gradiente estocastico). O lbfgs da LogisticRegression
    # materializa a matriz inteira e estourou a RAM desta maquina com o dataset
    # aumentado; o SGD nunca segura tudo de uma vez. Rodamos os DOIS bracos
    # (sem/com aug) com ele para a comparacao ser justa.
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, tol=1e-3, random_state=0),
    )


def _logo_com_aug(world, rotulo, grupos, *, aug: bool) -> np.ndarray:
    """LeaveOneGroupOut. Se aug=True, aumenta o treino de cada dobra. Devolve
    predicoes fora-da-dobra."""
    logo = LeaveOneGroupOut()
    y_pred = np.empty_like(rotulo)

    for tr, te in logo.split(world, rotulo, grupos):
        w_tr, y_tr = (world[tr], rotulo[tr])
        if aug:
            # 4x (original + espelho + 1 rotacao de cada). A maquina esta com
            # pouca RAM livre; 6x estourou. 4x ja cobre pose + lateralidade.
            w_tr, y_tr = aumentar_treino(w_tr, y_tr, n_rotacoes=1, espelhar=True)

        X_tr, X_te = _features(w_tr), _features(world[te])
        modelo = _pipe().fit(X_tr, y_tr)
        y_pred[te] = modelo.predict(X_te)

        # Libera explicitamente entre dobras -- a maquina esta no limite de RAM.
        del w_tr, y_tr, X_tr, X_te, modelo
        gc.collect()

    return y_pred


def _acc_por_sessao(rotulo, y_pred, grupos) -> tuple[float, float]:
    sessoes = sorted(set(grupos.tolist()))
    accs = [accuracy_score(rotulo[grupos == s], y_pred[grupos == s]) for s in sessoes]
    return float(np.mean(accs)), float(min(accs))


def _par_entre_sessoes(world, rotulo, grupos, par: list[str], *, aug: bool) -> float:
    """Acuracia media entre sessoes so para as 2 letras do par."""
    m = np.isin(rotulo, par)
    y_pred = _logo_com_aug(world[m], rotulo[m], grupos[m], aug=aug)
    sessoes = sorted(set(grupos[m].tolist()))
    return float(
        np.mean(
            [accuracy_score(rotulo[m][grupos[m] == s], y_pred[grupos[m] == s]) for s in sessoes]
        )
    )


def main() -> int:
    ds = carregar_dataset()
    world, rotulo, grupos = ds.world, ds.rotulo, ds.grupos
    print(
        f"cru: {world.shape[0]} amostras, {len(ds.rotulos_unicos)} letras, "
        f"{len(ds.sessoes_unicas)} sessoes  (classificador: SGDClassifier log_loss)"
    )

    # OS PARES vem PRIMEIRO: sao pequenos (2 letras) e sao o teste real do
    # diagnostico. O run das 20 letras com aug e grande e a maquina esta sem RAM.
    print("\n" + "=" * 62)
    print("PARES CONFUSOS entre sessoes -- o teste do diagnostico")
    print("=" * 62)
    print(f"  {'par':<8} {'sem aug':>9} {'com aug':>9}   veredito")
    print("  " + "-" * 44)
    for par in (["T", "F"], ["R", "U"], ["U", "V"]):
        sem = _par_entre_sessoes(world, rotulo, grupos, par, aug=False)
        com = _par_entre_sessoes(world, rotulo, grupos, par, aug=True)
        delta = com - sem
        veredito = "AJUDOU" if delta > 0.03 else ("piorou" if delta < -0.03 else "~igual")
        print(f"  {'/'.join(par):<8} {sem:>8.1%} {com:>8.1%}   {veredito} ({delta:+.1%})")

    print("\n" + "=" * 62)
    print("20 LETRAS -- sem vs com augmentation (LeaveOneGroupOut)")
    print("=" * 62)
    yp = _logo_com_aug(world, rotulo, grupos, aug=False)
    media, pior = _acc_por_sessao(rotulo, yp, grupos)
    print(f"  {'sem aug':<10} media entre sessoes {media:.1%}   sessao dificil {pior:.1%}")
    try:
        yp = _logo_com_aug(world, rotulo, grupos, aug=True)
        media, pior = _acc_por_sessao(rotulo, yp, grupos)
        print(f"  {'com aug':<10} media entre sessoes {media:.1%}   sessao dificil {pior:.1%}")
    except MemoryError:
        print("  com aug    [PULADO] a maquina ficou sem RAM (veja os pares acima,")
        print("             que ja respondem o diagnostico).")

    print("\nInterpretacao: se os pares AJUDARAM, o gap entre sessoes era de POSE")
    print("(inclinacao) -> augmentation resolve. Se nao, e inconsistencia de")
    print("EXECUCAO -> so mais sessoes resolvem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
