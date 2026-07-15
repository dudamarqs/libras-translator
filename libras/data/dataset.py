"""Armazenamento do dataset: salvar e carregar sessoes de coleta.

TRES decisoes de engenharia de dados moram aqui, e cada uma evita um erro que
custa caro depois:

1. GUARDAMOS LANDMARKS CRUS, NAO AS FEATURES JA NORMALIZADAS.
   Coletar de novo e caro (precisa de voce na frente da camera). Recalcular
   features e gratis. Se amanha a `features.py` melhorar, com os dados crus a
   gente REPROCESSA; se tivessemos salvo so as features, teriamos que
   RECOLETAR. Regra: guarde perto do cru, derive as features na hora (Etapa 7).

2. CADA SESSAO E UM GRUPO, E ISSO FICA GRAVADO.
   A 30 FPS, frames vizinhos sao quase identicos. Se o split treino/teste os
   separar aleatoriamente, o teste vira copia do treino e a acuracia medida e
   mentira (data leakage). A defesa: o split e feito por SESSAO inteira. Para
   isso, cada amostra carrega o id da sua sessao -- e este modulo garante isso.

3. FORMATO INSPECIONAVEL E AUTODESCRITIVO.
   Cada sessao = uma pasta com `dados.npz` (os arrays) + `meta.json` (quem,
   quando, com que camera, qual commit de codigo). Daqui a seis meses voce
   ainda consegue saber como cada dado nasceu.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from libras.vision.hands import N_LANDMARKS

RAIZ_PADRAO = Path(__file__).resolve().parents[2] / "datasets" / "raw"

# Versao do formato em disco. Se algum dia mudarmos o layout do npz, este numero
# muda e o carregador sabe distinguir sessoes antigas de novas -- em vez de
# quebrar com um erro obscuro sobre uma chave que nao existe.
FORMATO_VERSAO = 1


@dataclass(slots=True)
class Sessao:
    """Uma sessao de coleta em memoria, antes de ir para o disco.

    Arrays paralelos (mesmo comprimento N = numero de amostras):
      world  : (N, 21, 3) landmarks metricos, em metros -- a MATERIA-PRIMA.
      imagem : (N, 21, 3) landmarks normalizados pela imagem -- guardados para
               depuracao/redesenho; NAO sao a feature.
      lado   : (N,) "Left"/"Right" -- a mao REAL (ja corrigida, ADR-011).
      rotulo : (N,) o rotulo de cada amostra (a letra).
    """

    session_id: str
    world: np.ndarray
    imagem: np.ndarray
    lado: np.ndarray
    rotulo: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.rotulo)
        if not (len(self.world) == len(self.imagem) == len(self.lado) == n):
            raise ValueError("arrays da sessao tem comprimentos diferentes")
        if n > 0 and self.world.shape[1:] != (N_LANDMARKS, 3):
            raise ValueError(f"world deve ser (N, {N_LANDMARKS}, 3), e {self.world.shape}")

    @property
    def n_amostras(self) -> int:
        return len(self.rotulo)

    def contagem(self) -> dict[str, int]:
        return dict(Counter(self.rotulo.tolist()))


def novo_session_id() -> str:
    # Ordenavel e legivel: sessao_20260715_143005. Ordenar por nome = ordenar
    # por tempo, o que ajuda a pegar "as N sessoes mais recentes" sem parsear
    # data nenhuma.
    return "sessao_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def salvar_sessao(sessao: Sessao, meta_extra: dict | None = None, raiz: Path | None = None) -> Path:
    """Grava a sessao em datasets/raw/<session_id>/."""
    raiz = raiz or RAIZ_PADRAO
    destino = raiz / sessao.session_id
    destino.mkdir(parents=True, exist_ok=True)

    # savez_compressed: landmarks sao muito redundantes (frames vizinhos quase
    # iguais) e comprimem otimo. Espaco em disco de graca.
    np.savez_compressed(
        destino / "dados.npz",
        world=sessao.world.astype(np.float32),
        imagem=sessao.imagem.astype(np.float32),
        lado=sessao.lado.astype("<U5"),
        rotulo=sessao.rotulo.astype("<U16"),
    )

    meta = {
        "formato_versao": FORMATO_VERSAO,
        "session_id": sessao.session_id,
        "criado_em": datetime.now(UTC).isoformat(),
        "n_amostras": sessao.n_amostras,
        "contagem_por_rotulo": sessao.contagem(),
        **(meta_extra or {}),
    }
    (destino / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    return destino


def carregar_sessao(pasta: Path) -> Sessao:
    dados = np.load(pasta / "dados.npz", allow_pickle=False)
    return Sessao(
        session_id=pasta.name,
        world=dados["world"],
        imagem=dados["imagem"],
        lado=dados["lado"],
        rotulo=dados["rotulo"],
    )


@dataclass(slots=True)
class DatasetBruto:
    """Todas as sessoes concatenadas, prontas para o pre-processamento.

    `grupos` e o vetor que torna o split honesto: grupos[i] e o session_id da
    amostra i. Qualquer split que respeite esses grupos (nenhuma sessao nos dois
    lados) esta livre do data leakage descrito no topo do modulo.
    """

    world: np.ndarray  # (N, 21, 3)
    imagem: np.ndarray  # (N, 21, 3)
    lado: np.ndarray  # (N,)
    rotulo: np.ndarray  # (N,)
    grupos: np.ndarray  # (N,) session_id de cada amostra

    @property
    def n_amostras(self) -> int:
        return len(self.rotulo)

    @property
    def rotulos_unicos(self) -> list[str]:
        return sorted(set(self.rotulo.tolist()))

    @property
    def sessoes_unicas(self) -> list[str]:
        return sorted(set(self.grupos.tolist()))

    def contagem_por_rotulo(self) -> dict[str, int]:
        return dict(sorted(Counter(self.rotulo.tolist()).items()))


def carregar_dataset(raiz: Path | None = None) -> DatasetBruto:
    """Le TODAS as sessoes de datasets/raw/ e concatena.

    Uma pasta sem `dados.npz` e ignorada (pode ser lixo, um .gitkeep, uma coleta
    interrompida). Preferimos pular a quebrar: perder uma sessao ruim e melhor
    do que impedir o treino por causa dela.
    """
    raiz = raiz or RAIZ_PADRAO
    pastas = sorted(p for p in raiz.iterdir() if p.is_dir() and (p / "dados.npz").exists())

    if not pastas:
        raise FileNotFoundError(
            f"nenhuma sessao encontrada em {raiz}.\n"
            "Rode:  .venv\\Scripts\\python.exe training\\collect.py"
        )

    sessoes = [carregar_sessao(p) for p in pastas]

    return DatasetBruto(
        world=np.concatenate([s.world for s in sessoes]),
        imagem=np.concatenate([s.imagem for s in sessoes]),
        lado=np.concatenate([s.lado for s in sessoes]),
        rotulo=np.concatenate([s.rotulo for s in sessoes]),
        grupos=np.concatenate(
            [np.full(s.n_amostras, s.session_id, dtype=object) for s in sessoes]
        ).astype("<U32"),
    )
