"""Treino e avaliacao da MLP -- Etapa 9.

    .venv\\Scripts\\python.exe training\\train_mlp.py

Usa a MESMA validacao do baseline (LeaveOneGroupOut por sessao, ADR-015) para
que os numeros sejam comparaveis maca-com-maca. A pergunta que este script
responde: a rede bate o LogReg (88% na sessao dificil) e desfaz os clusters
R/U/V, T/F, M/N da matriz de confusao?

Estrutura:
  1. DEMO DE APRENDIZADO -- treina uma vez mostrando a perda cair e a diferenca
     entre acuracia de treino e de teste (evidencia de overfitting).
  2. AVALIACAO HONESTA   -- LeaveOneGroupOut completo, acuracia por sessao.
  3. MATRIZ DE CONFUSAO + comparacao direta dos clusters com o baseline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from libras.models.mlp import prever_indices, treinar_rede

FEATURES = Path(__file__).resolve().parent.parent / "datasets" / "processed" / "features.npz"

# Numeros honestos do baseline LogReg (20 letras), para comparar.
BASELINE_MEDIA = 0.959
BASELINE_SESSAO_DIFICIL = 0.884

# Clusters que o baseline confundiu (verdade -> predicao : contagem no baseline).
CLUSTERS_BASELINE = {
    ("R", "U"): 119, ("U", "R"): 72, ("U", "V"): 34, ("R", "G"): 53,
    ("T", "F"): 104, ("F", "T"): 27, ("M", "N"): 9, ("N", "M"): 18,
    ("D", "C"): 19, ("C", "F"): 18, ("C", "O"): 12, ("S", "A"): 15,
}  # fmt: skip


def _carregar() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    dados = np.load(FEATURES, allow_pickle=False)
    X, y, grupos = dados["X"], dados["y"], dados["grupos"]
    classes = sorted(set(y.tolist()))
    return X, y, grupos, classes


def _para_indices(y: np.ndarray, classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[v] for v in y], dtype=np.int64)


def demo_aprendizado(X, y_idx, grupos, classes) -> None:
    """Treina uma vez mostrando a perda cair -- e o overfitting aparecer."""
    print("\n" + "=" * 60)
    print("1. DEMO DE APRENDIZADO (teste = a sessao mais dificil)")
    print("=" * 60)

    # Segura a sessao mais recente (o alfabeto completo, a mais dificil).
    sessao_teste = sorted(set(grupos.tolist()))[-1]
    treino = grupos != sessao_teste
    teste = grupos == sessao_teste

    # StandardScaler AJUSTADO SO NO TREINO. Ajustar no dataset todo deixaria o
    # teste "vazar" estatisticas para o treino -- uma forma sutil de leakage.
    scaler = StandardScaler().fit(X[treino])
    Xtr, Xte = scaler.transform(X[treino]), scaler.transform(X[teste])

    def mostrar(epoca: int, perda: float) -> None:
        if epoca % 40 == 0 or epoca == 199:
            print(f"  epoca {epoca:3d}   perda de treino: {perda:.4f}")

    print(f"  treinando (teste sera a sessao {sessao_teste})...")
    modelo = treinar_rede(Xtr, y_idx[treino], n_classes=len(classes), epocas=200, monitorar=mostrar)

    acc_treino = accuracy_score(y_idx[treino], prever_indices(modelo, Xtr))
    acc_teste = accuracy_score(y_idx[teste], prever_indices(modelo, Xte))
    print(f"\n  acuracia no TREINO: {acc_treino:.1%}")
    print(f"  acuracia no TESTE : {acc_teste:.1%}")
    gap = acc_treino - acc_teste
    print(
        f"  diferenca (gap): {gap:.1%}  -> "
        + (
            "sinal de overfitting: a rede decora melhor do que generaliza."
            if gap > 0.08
            else "gap pequeno: a rede esta generalizando bem."
        )
    )


def avaliar_logo(X, y, y_idx, grupos, classes) -> np.ndarray:
    """LeaveOneGroupOut completo. Devolve as predicoes fora-da-dobra (strings)."""
    print("\n" + "=" * 60)
    print("2. AVALIACAO HONESTA (LeaveOneGroupOut por sessao)")
    print("=" * 60)

    logo = LeaveOneGroupOut()
    y_pred_idx = np.empty_like(y_idx)

    for treino_idx, teste_idx in logo.split(X, y_idx, grupos):
        sessao = grupos[teste_idx][0]
        scaler = StandardScaler().fit(X[treino_idx])
        Xtr = scaler.transform(X[treino_idx])
        Xte = scaler.transform(X[teste_idx])

        modelo = treinar_rede(Xtr, y_idx[treino_idx], n_classes=len(classes), epocas=200)
        pred = prever_indices(modelo, Xte)
        y_pred_idx[teste_idx] = pred

        acc = accuracy_score(y_idx[teste_idx], pred)
        print(f"  teste = {sessao}    acc = {acc:6.1%}  (n={len(teste_idx)})")

    accs = [
        accuracy_score(y[grupos == s], np.array(classes)[y_pred_idx][grupos == s])
        for s in sorted(set(grupos.tolist()))
    ]
    print(f"\n  media entre sessoes: {np.mean(accs):.1%}  (desvio {np.std(accs):.1%})")
    return np.array(classes)[y_pred_idx]


def _matriz_confusao(y, y_pred, classes) -> None:
    print("\n" + "=" * 60)
    print("3. MATRIZ DE CONFUSAO DA MLP (linha=verdade, coluna=predicao)")
    print("=" * 60)
    cm = confusion_matrix(y, y_pred, labels=classes)
    print("        " + "".join(f"{c:>6}" for c in classes))
    for i, c in enumerate(classes):
        print(f"  {c:>4}  " + "".join(f"{cm[i, j]:>6}" for j in range(len(classes))))


def _comparar_clusters(y, y_pred) -> None:
    print("\n" + "=" * 60)
    print("CLUSTERS PROBLEMATICOS: baseline LogReg  ->  MLP")
    print("=" * 60)
    print(f"  {'confusao':<12} {'LogReg':>8} {'MLP':>8}   veredito")
    print("  " + "-" * 44)
    melhoras = 0
    for (verdade, pred), n_base in CLUSTERS_BASELINE.items():
        n_mlp = int(np.sum((y == verdade) & (y_pred == pred)))
        veredito = "melhorou" if n_mlp < n_base else ("igual" if n_mlp == n_base else "PIOROU")
        melhoras += n_mlp < n_base
        print(f"  {verdade}->{pred:<9} {n_base:>8} {n_mlp:>8}   {veredito}")
    print(f"\n  {melhoras}/{len(CLUSTERS_BASELINE)} confusoes melhoraram com a MLP.")


def main() -> int:
    X, y, grupos, classes = _carregar()
    y_idx = _para_indices(y, classes)
    print(f"X: {X.shape}   classes: {len(classes)}   sessoes: {len(set(grupos.tolist()))}")

    demo_aprendizado(X, y_idx, grupos, classes)
    y_pred = avaliar_logo(X, y, y_idx, grupos, classes)
    _matriz_confusao(y, y_pred, classes)
    _comparar_clusters(y, y_pred)

    print("\n" + "=" * 60)
    acc_mlp = accuracy_score(y, y_pred)
    print("VEREDITO FINAL (numeros honestos, 20 letras)")
    print("=" * 60)
    print(f"  baseline LogReg : media {BASELINE_MEDIA:.1%}")
    print(f"  baseline (sessao dificil) : {BASELINE_SESSAO_DIFICIL:.1%}")
    print(f"  MLP (global)    : {acc_mlp:.1%}")
    print("  (compare a media-entre-sessoes acima com a do baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
