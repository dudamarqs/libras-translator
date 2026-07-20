"""Etapa 8b -- treina o modelo FINAL e o exporta para producao.

    .venv\\Scripts\\python.exe training\\exportar.py

Diferenca para o train.py (a Etapa 8): o train.py MEDE, segurando uma sessao por
vez (LeaveOneGroupOut) para estimar o quanto o modelo generaliza. Este script
PRODUZ o artefato: treina o mesmo pipeline em TODAS as sessoes e salva em
models/datilologia/. Estimar e servir sao passos diferentes -- e o modelo servido
usa cada amostra que existe.

Registramos no meta.json a metrica HONESTA (a mesma LeaveOneGroupOut do train.py),
para o artefato carregar consigo a nota em que da para confiar -- e nao o numero
otimista de reclassificar o proprio treino.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

from libras.models.classificador import Classificador, construir_pipeline

FEATURES = Path(__file__).resolve().parent.parent / "datasets" / "processed" / "features.npz"


def main() -> int:
    if not FEATURES.exists():
        print(f"{FEATURES} nao existe. Rode antes:  python training\\preprocess.py")
        return 1

    dados = np.load(FEATURES, allow_pickle=False)
    X, y, grupos = dados["X"], dados["y"], dados["grupos"]
    sessoes = sorted(set(grupos.tolist()))
    print(f"X: {X.shape}   classes: {sorted(set(y.tolist()))}   sessoes: {len(sessoes)}")

    # Metrica honesta para o registro -- a MESMA validacao da Etapa 8.
    scores = cross_val_score(
        construir_pipeline(), X, y, groups=grupos, cv=LeaveOneGroupOut()
    )
    acc_media, acc_desvio = float(np.mean(scores)), float(np.std(scores))
    print(f"acuracia honesta (LeaveOneGroupOut): {acc_media:.1%}  (desvio {acc_desvio:.1%})")

    # Modelo final: treina em TUDO.
    modelo = Classificador.treinar(X, y)
    destino = modelo.salvar(
        meta_extra={
            "n_amostras_treino": int(len(y)),
            "sessoes": sessoes,
            "acuracia_logo_media": round(acc_media, 4),
            "acuracia_logo_desvio": round(acc_desvio, 4),
            "validacao": "LeaveOneGroupOut por sessao",
            "aviso": (
                "3 sessoes da MESMA pessoa/mao: generaliza entre luz/posicao, "
                "nao entre pessoas. Datilologia estatica; H J K X Y Z sao dinamicas."
            ),
        }
    )
    print(f"\nModelo salvo em: {destino}")
    print("Agora rode a demo:  .venv\\Scripts\\python.exe scripts\\reconhecer.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
