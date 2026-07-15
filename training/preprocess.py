"""Pre-processamento: landmarks crus -> matriz de features pronta para treino.

    .venv\\Scripts\\python.exe training\\preprocess.py

O que faz: le TODAS as sessoes de datasets/raw/, aplica a normalizacao de
features (a MESMA funcao que a inferencia usa -- `normalizar_mao`), e salva a
matriz X, os rotulos y e os GRUPOS (a que sessao cada amostra pertence) em
datasets/processed/features.npz.

Por que este passo existe como script separado (e nao embutido no treino):

  - Extrair features e deterministico e um pouco lento; separar deixa voce
    treinar varios modelos sobre a MESMA matriz sem recomputar.
  - E o ponto onde o cru vira feature, aplicando `libras.vision.features`. Se
    um dia a normalizacao mudar, voce roda ISTO de novo -- sem recoletar nada
    (ADR-013). Os dados crus continuam intactos em datasets/raw/.

IMPORTANTE -- por que NAO fazemos o split treino/teste aqui:
  O split e responsabilidade do TREINO (Etapa 8), e ele e feito por SESSAO
  (GroupShuffleSplit sobre `grupos`) para evitar data leakage. Guardamos os
  grupos junto das features justamente para que o treino possa fazer isso.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from libras.data.dataset import carregar_dataset
from libras.vision.features import N_FEATURES_MAO, normalizar_mao

DESTINO_PADRAO = Path(__file__).resolve().parent.parent / "datasets" / "processed"


def _commit_atual() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "desconhecido"


def main() -> int:
    print("\nCarregando dataset cru...")
    ds = carregar_dataset()

    print(
        f"  {ds.n_amostras} amostras, {len(ds.rotulos_unicos)} letras, "
        f"{len(ds.sessoes_unicas)} sessao(oes)"
    )

    # cru -> features. UMA mao (datilologia): 63 numeros, o lado e ignorado
    # (ver ADR-014). normalizar_mao e a mesma funcao chamada na inferencia.
    print("Extraindo features (uma mao, 63 dims)...")
    X = np.array([normalizar_mao(w) for w in ds.world], dtype=np.float32)
    y = ds.rotulo.astype("<U16")
    grupos = ds.grupos.astype("<U32")

    assert X.shape == (ds.n_amostras, N_FEATURES_MAO), X.shape

    # Sanidade: features envenenadas matam o treino silenciosamente.
    n_nan = int(np.isnan(X).any(axis=1).sum())
    n_zero = int(np.all(X == 0, axis=1).sum())
    if n_nan:
        print(f"  [ATENCAO] {n_nan} amostras com NaN -- serao um problema no treino")
    if n_zero:
        print(f"  [info] {n_zero} amostras degeneradas (tudo zero, mao mal detectada)")

    DESTINO_PADRAO.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DESTINO_PADRAO / "features.npz", X=X, y=y, grupos=grupos)

    classes = sorted(set(y.tolist()))
    meta = {
        "criado_em": datetime.now(UTC).isoformat(),
        "commit": _commit_atual(),
        "n_amostras": int(ds.n_amostras),
        "n_features": N_FEATURES_MAO,
        "representacao": "uma_mao_63d",
        "classes": classes,
        "contagem_por_classe": dict(sorted(Counter(y.tolist()).items())),
        "sessoes": ds.sessoes_unicas,
    }
    (DESTINO_PADRAO / "features_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), "utf-8"
    )

    print(f"\nSalvo em {DESTINO_PADRAO / 'features.npz'}")
    print(f"  X: {X.shape}   classes: {classes}")

    # O AVISO QUE IMPORTA. Split por sessao precisa de >= 2 sessoes; com uma so,
    # nao existe conjunto de teste honesto (todo frame de teste teria um quase-
    # gemeo no treino). O treino ainda roda, mas a acuracia sera OTIMISTA.
    if len(ds.sessoes_unicas) < 2:
        print(
            "\n" + "!" * 70 + "\n"
            "So ha 1 SESSAO. Da para treinar, mas a acuracia medida sera OTIMISTA:\n"
            "frames da mesma sessao sao quase identicos, entao qualquer teste tirado\n"
            "dela e praticamente uma copia do treino (data leakage -- ADR-013).\n"
            "Para um numero em que da pra confiar, colete 2-3 sessoes, em dias/luzes\n"
            "diferentes, e rode este script de novo.\n" + "!" * 70
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
