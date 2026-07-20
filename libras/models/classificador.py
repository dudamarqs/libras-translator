"""O classificador da datilologia -- definicao, predicao e persistencia.

Este modulo carrega DUAS responsabilidades que precisam morar juntas:

1. A DEFINICAO do modelo de producao (`construir_pipeline`). E a MESMA que a
   Etapa 8 avaliou -- o LogReg que venceu com 99.7% entre sessoes. Ela fica aqui,
   num lugar so, para que o modelo que voce MEDE seja exatamente o modelo que
   voce SERVE. Se a Etapa 8 avaliasse um pipeline e a producao servisse outro, a
   nota de corte seria uma ficcao (e a versao mais sutil do skew do ADR-004).

2. A INFERENCIA em tempo real (`Classificador.prever`). Recebe o vetor de 63
   numeros (a mesma saida de `features.vetor_uma_mao` que o treino consumiu) e
   devolve letra + confianca.

Sobre a CONFIANCA: um classificador com probabilidade SEMPRE devolve uma classe,
mesmo quando voce nao faz sinal nenhum. Ele nao sabe dizer "nao sei" -- distribui
probabilidade entre as letras que conhece e escolhe a maior. Por isso `prever`
devolve a confianca junto: e quem chama (a UI) que decide o limiar de aceitacao.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from libras.vision.features import N_FEATURES_MAO

# Quantos vizinhos de treino olhar para medir "novidade". Media de k suaviza o
# efeito de um unico ponto de treino por acaso proximo.
K_VIZINHOS = 5

DIR_PADRAO = Path(__file__).resolve().parents[2] / "models" / "datilologia"
ARQ_MODELO = "modelo.joblib"
ARQ_META = "meta.json"


def construir_pipeline() -> Pipeline:
    """O modelo de PRODUCAO da datilologia (ver cabecalho do modulo).

    StandardScaler + LogisticRegression: o scaler faz parte do modelo (nao um
    passo solto), entao ele viaja junto no arquivo salvo e a inferencia aplica
    exatamente a mesma transformacao do treino, sem ninguem precisar lembrar.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def _commit_atual() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "desconhecido"


@dataclass(slots=True, frozen=True)
class Predicao:
    """O que o classificador devolve por frame."""

    letra: str
    confianca: float  # 0..1, a probabilidade da classe escolhida
    distancia: float  # distancia media aos K vizinhos de treino (espaco escalado)
    desconhecido: bool  # True = longe demais de tudo que o modelo viu; NAO chute


class Classificador:
    """Envolve o pipeline treinado e traduz vetor -> Predicao.

    Alem do pipeline (scaler + LogReg), guarda um detector de NOVIDADE: um
    NearestNeighbors ajustado nos dados de treino (ja escalados). A softmax do
    LogReg diz "qual das 5 letras" -- mas nao sabe dizer "isso nao e nenhuma
    letra". Um classificador fechado e cego para o que nunca viu, e -- pior --
    fica MAIS confiante quanto mais longe da fronteira o ponto cai (foi o C de
    100% na mao coçando a cabeça). A distancia ao treino e a peca que devolve ao
    sistema o direito de dizer "nao sei": longe de tudo => desconhecido.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        classes: list[str],
        nn: NearestNeighbors,
        limiar_novidade: float,
    ) -> None:
        self._pipeline = pipeline
        self._classes = list(classes)
        self._nn = nn
        self._limiar_novidade = float(limiar_novidade)

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    @property
    def limiar_novidade(self) -> float:
        return self._limiar_novidade

    @classmethod
    def treinar(cls, X: np.ndarray, y: np.ndarray) -> Classificador:
        """Ajusta o modelo de producao em TODOS os dados dados.

        Note que aqui NAO ha split: a validacao honesta (quanto ele generaliza)
        e trabalho da Etapa 8, feito por sessao. O modelo que vai para producao
        usa cada amostra que existe -- jogar 1/3 fora aqui so o deixaria pior sem
        medir nada.

        Calibra tambem o LIMIAR DE NOVIDADE a partir dos proprios dados: quao
        longe uma amostra normal fica dos seus vizinhos de treino? Tudo muito
        alem disso e considerado "nao visto". Sem numero magico -- sai da
        distribuicao real (ver `_calibrar_novidade`).
        """
        pipeline = construir_pipeline()
        pipeline.fit(X, y)

        # Espaco ESCALADO (a saida do scaler, entrada do LogReg). E nele que as
        # distancias fazem sentido: cada feature com a mesma importancia.
        X_escalado = pipeline[:-1].transform(X)
        nn = NearestNeighbors(n_neighbors=K_VIZINHOS + 1).fit(X_escalado)
        limiar = cls._calibrar_novidade(nn, X_escalado)

        return cls(pipeline, list(pipeline.classes_), nn, limiar)

    @staticmethod
    def _calibrar_novidade(nn: NearestNeighbors, X_escalado: np.ndarray) -> float:
        """Distancia tipica de uma amostra normal aos seus vizinhos de treino.

        Para cada ponto de treino, a media da distancia aos K vizinhos (pulando
        o vizinho 0, que e ele mesmo, a distancia zero). O limiar e o percentil
        99 dessa distribuicao, com uma folga -- um ponto NOVO de verdade (uma mao
        que nao e letra) cai muito alem disso.
        """
        dists, _ = nn.kneighbors(X_escalado)  # (N, K+1); coluna 0 = ele mesmo
        d_por_amostra = dists[:, 1:].mean(axis=1)
        return float(np.percentile(d_por_amostra, 99) * 1.5)

    def prever(self, vetor: np.ndarray) -> Predicao:
        """(63,) -> Predicao. O vetor e a saida de `features.vetor_uma_mao`."""
        if vetor.shape != (N_FEATURES_MAO,):
            raise ValueError(f"esperado vetor ({N_FEATURES_MAO},), recebido {vetor.shape}")

        x = vetor.reshape(1, -1)
        probas = self._pipeline.predict_proba(x)[0]
        i = int(np.argmax(probas))

        # Novidade: distancia media aos K vizinhos de treino, no espaco escalado.
        # Aqui o ponto NAO esta no treino, entao todos os K sao vizinhos de fato.
        x_escalado = self._pipeline[:-1].transform(x)
        dists, _ = self._nn.kneighbors(x_escalado, n_neighbors=K_VIZINHOS)
        distancia = float(dists.mean())

        return Predicao(
            letra=self._classes[i],
            confianca=float(probas[i]),
            distancia=distancia,
            desconhecido=distancia > self._limiar_novidade,
        )

    # -- persistencia ------------------------------------------------------

    def salvar(self, destino: Path | None = None, meta_extra: dict | None = None) -> Path:
        """Grava modelo.joblib + meta.json em models/datilologia/.

        O `meta.json` e legivel por humanos e responde, seis meses depois: que
        letras este modelo conhece? com que dados nasceu? qual commit? qual
        versao do sklearn (o pickle do joblib depende dela para recarregar)?
        """
        destino = destino or DIR_PADRAO
        destino.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "pipeline": self._pipeline,
                "classes": self._classes,
                "nn": self._nn,
                "limiar_novidade": self._limiar_novidade,
            },
            destino / ARQ_MODELO,
        )

        meta = {
            "criado_em": datetime.now(UTC).isoformat(),
            "commit": _commit_atual(),
            "sklearn": sklearn.__version__,
            "representacao": "uma_mao_63d",
            "n_features": N_FEATURES_MAO,
            "classes": self._classes,
            "limiar_novidade": round(self._limiar_novidade, 4),
            **(meta_extra or {}),
        }
        (destino / ARQ_META).write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
        return destino

    @classmethod
    def carregar(cls, origem: Path | None = None) -> Classificador:
        origem = origem or DIR_PADRAO
        arquivo = origem / ARQ_MODELO
        if not arquivo.exists():
            raise FileNotFoundError(
                f"modelo nao encontrado em {arquivo}\n"
                "Treine e exporte antes:  .venv\\Scripts\\python.exe training\\exportar.py"
            )
        dados = joblib.load(arquivo)
        return cls(
            dados["pipeline"],
            dados["classes"],
            dados["nn"],
            dados["limiar_novidade"],
        )
