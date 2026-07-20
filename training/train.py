"""Etapa 8 -- BASELINE do classificador (scikit-learn).

    .venv\\Scripts\\python.exe training\\train.py

O que faz: le datasets/processed/features.npz (X, y, grupos) e mede, de forma
HONESTA, quao bem modelos classicos separam as letras. Nao salva modelo nenhum
ainda -- a funcao desta etapa e produzir UM NUMERO em que da pra confiar, a nota
de corte que a rede neural (Etapa 9) tera que bater para justificar sua
existencia (ADR-002: baseline obrigatorio antes de qualquer rede).

------------------------------------------------------------------------------
POR QUE LeaveOneGroupOut, E NAO um split aleatorio
------------------------------------------------------------------------------
A 30 FPS, frames vizinhos da MESMA sessao sao quase identicos. Um split
aleatorio jogaria copias quase-gemeas nos dois lados (treino e teste) e a
acuracia mediria a capacidade do modelo de RECONHECER o que ja viu -- nao de
GENERALIZAR. Foi isso que inflou o 99.8% da primeira medicao (data leakage,
ADR-013).

A defesa e treinar e testar em sessoes DIFERENTES. Com 3 sessoes,
LeaveOneGroupOut e a escolha certa: 3 dobras, cada sessao e o teste exatamente
uma vez, treinando nas outras duas. E DETERMINISTICO (nada de sorteio), entao o
numero e reproduzivel e comparavel entre execucoes -- diferente do
GroupShuffleSplit que o roteiro previa. Ver ADR-015.

Cada dobra responde: "treinei em 2 condicoes de luz, acerto numa TERCEIRA que o
modelo nunca viu?". E exatamente a pergunta do mundo real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

from libras.models.classificador import construir_pipeline

FEATURES = Path(__file__).resolve().parent.parent / "datasets" / "processed" / "features.npz"


def _carregar() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not FEATURES.exists():
        raise FileNotFoundError(
            f"{FEATURES} nao existe.\n"
            "Rode antes:  .venv\\Scripts\\python.exe training\\preprocess.py"
        )
    dados = np.load(FEATURES, allow_pickle=False)
    return dados["X"], dados["y"], dados["grupos"]


def _modelos() -> dict[str, object]:
    """Os dois baselines classicos.

    LogReg entra num Pipeline com StandardScaler: modelos lineares sao sensiveis
    a escala das features, e -- crucial -- o scaler e AJUSTADO dentro de cada
    dobra de treino, nunca vendo o teste. Por isso ele mora no pipeline, e nao
    aplicado uma vez sobre X inteiro (isso vazaria estatisticas do teste).

    RandomForest nao precisa de escala (decide por limiares), entao vai cru.
    """
    return {
        # A MESMA definicao que vai para producao (libras.models.classificador).
        # Avaliar aqui exatamente o pipeline que sera servido evita o skew do
        # ADR-004 na sua forma mais sutil: medir um modelo e servir outro.
        "LogReg": construir_pipeline(),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=0),
    }


def _avaliar(nome: str, modelo, X: np.ndarray, y: np.ndarray, grupos: np.ndarray) -> np.ndarray:
    """Roda LeaveOneGroupOut e devolve as predicoes fora-da-dobra (uma por amostra).

    Cada amostra e prevista pelo modelo que NAO a viu no treino -- entao juntar
    todas as predicoes forma um retrato honesto do sistema inteiro.
    """
    logo = LeaveOneGroupOut()
    y_pred = np.empty_like(y)
    sessoes = sorted(set(grupos.tolist()))

    print(f"\n{'=' * 60}\n{nome}\n{'=' * 60}")
    for treino_idx, teste_idx in logo.split(X, y, grupos):
        sessao_teste = grupos[teste_idx][0]
        modelo.fit(X[treino_idx], y[treino_idx])
        pred = modelo.predict(X[teste_idx])
        y_pred[teste_idx] = pred
        acc = accuracy_score(y[teste_idx], pred)
        # Alinha o nome da sessao para a coluna de acuracia ficar legivel.
        print(f"  teste = {sessao_teste:<24}  acc = {acc:6.1%}  (n={len(teste_idx)})")

    accs = [
        accuracy_score(y[grupos == s], y_pred[grupos == s]) for s in sessoes
    ]
    print(f"\n  media entre sessoes: {np.mean(accs):.1%}  (desvio {np.std(accs):.1%})")
    return y_pred


def _matriz_confusao(y: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> None:
    """Matriz de confusao em texto. Linha = verdade, coluna = predicao.

    A diagonal e o acerto; tudo fora dela e uma confusao especifica -- e e ai
    que mora a informacao. (Previsao registrada no ADR-014: se algo se confunde,
    aposto em C<->O, as duas maos redondas.)
    """
    cm = confusion_matrix(y, y_pred, labels=classes)
    print(f"\n{'=' * 60}\nMATRIZ DE CONFUSAO (linha=verdade, coluna=predicao)\n{'=' * 60}")
    cabecalho = "        " + "".join(f"{c:>7}" for c in classes)
    print(cabecalho)
    for i, c in enumerate(classes):
        linha = "".join(f"{cm[i, j]:>7}" for j in range(len(classes)))
        print(f"  {c:<5}{linha}")


def main() -> int:
    X, y, grupos = _carregar()
    classes = sorted(set(y.tolist()))
    sessoes = sorted(set(grupos.tolist()))

    print(f"X: {X.shape}   classes: {classes}   sessoes: {len(sessoes)}")
    if len(sessoes) < 2:
        print("\n[ERRO] menos de 2 sessoes -- nao ha split honesto. Colete mais e reprocesse.")
        return 1

    # Chute puro (a classe majoritaria) e o piso absoluto: qualquer modelo que
    # nao supere isso nao aprendeu nada.
    _, contagens = np.unique(y, return_counts=True)
    baseline_burro = contagens.max() / len(y)
    print(f"chute majoritario (piso): {baseline_burro:.1%}")

    # Guardamos a predicao de cada modelo com sua acuracia media entre sessoes,
    # para a matriz de confusao retratar o VENCEDOR -- nao o ultimo a rodar.
    resultados: dict[str, tuple[float, np.ndarray]] = {}
    for nome, modelo in _modelos().items():
        pred = _avaliar(nome, modelo, X, y, grupos)
        acc_media = np.mean([accuracy_score(y[grupos == s], pred[grupos == s]) for s in sessoes])
        resultados[nome] = (float(acc_media), pred)

    vencedor = max(resultados, key=lambda n: resultados[n][0])
    print(f"\nMelhor baseline: {vencedor} ({resultados[vencedor][0]:.1%})")
    _matriz_confusao(y, resultados[vencedor][1], classes)
    print("\nBaseline concluido. Este e o numero que a rede (Etapa 9) tera que bater.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
