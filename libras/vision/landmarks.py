"""Estruturas e constantes PURAS dos landmarks -- sem dependencia do MediaPipe.

Por que este modulo existe (e foi extraido de hands.py):

Ler um dataset do disco NAO deveria carregar uma biblioteca de visao
computacional. Antes, `dataset.py` importava `N_LANDMARKS` de `hands.py`, e
`hands.py` importa o MediaPipe inteiro (centenas de MB de libs nativas). Numa
maquina com pouca RAM, isso chegou a falhar so de IMPORTAR ("paging file too
small") -- um treino que nao usa camera nenhuma sendo impedido por causa de um
`import` transitivo.

A regra: as ESTRUTURAS DE DADOS (o que uma mao E) vivem aqui, leves e puras. O
DETECTOR (o que depende do MediaPipe para PRODUZIR uma mao) fica em `hands.py`.
Quem so consome dados -- coleta, pre-processamento, treino -- importa daqui e
nunca toca no MediaPipe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_LANDMARKS = 21

# Os 21 pontos, na ordem em que o MediaPipe os devolve.
# Vale decorar a estrutura: PULSO + 5 dedos x 4 juntas.
# Cada dedo vai da base (CMC/MCP) ate a ponta (TIP).
NOMES_LANDMARKS: tuple[str, ...] = (
    "PULSO",
    "POLEGAR_CMC", "POLEGAR_MCP", "POLEGAR_IP", "POLEGAR_PONTA",
    "INDICADOR_MCP", "INDICADOR_PIP", "INDICADOR_DIP", "INDICADOR_PONTA",
    "MEDIO_MCP", "MEDIO_PIP", "MEDIO_DIP", "MEDIO_PONTA",
    "ANELAR_MCP", "ANELAR_PIP", "ANELAR_DIP", "ANELAR_PONTA",
    "MINIMO_MCP", "MINIMO_PIP", "MINIMO_DIP", "MINIMO_PONTA",
)  # fmt: skip

# O "esqueleto" da mao: quais pontos se ligam a quais. So para desenhar.
CONEXOES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),            # indicador
    (0, 9), (9, 10), (10, 11), (11, 12),       # medio
    (0, 13), (13, 14), (14, 15), (15, 16),     # anelar
    (0, 17), (17, 18), (18, 19), (19, 20),     # minimo
    (5, 9), (9, 13), (13, 17),                 # a palma, ligando as bases
)  # fmt: skip


_OPOSTO: dict[str, str] = {"Left": "Right", "Right": "Left"}


def _lado_real(lado_mediapipe: str, *, espelhado: bool) -> str:
    """Converte a lateralidade que o MediaPipe VIU na lateralidade REAL.

    Num frame espelhado, a mao direita do usuario tem a geometria de uma mao
    esquerda -- entao o MediaPipe responde "Left". Ele nao esta errado: esta
    descrevendo corretamente a imagem que recebeu. Quem tem que traduzir de
    volta para o mundo real somos nos.

    Se o frame NAO foi espelhado, nao ha o que corrigir.

    O `.get(..., lado)` em vez de `[...]`: se um dia o MediaPipe devolver uma
    categoria inesperada, preferimos passar o valor adiante a estourar um
    KeyError no meio do loop de captura. Um rotulo estranho e visivel no HUD;
    uma excecao derruba o sistema inteiro.
    """
    if not espelhado:
        return lado_mediapipe
    return _OPOSTO.get(lado_mediapipe, lado_mediapipe)


@dataclass(slots=True, frozen=True)
class Mao:
    """Uma mao detectada."""

    lado: str
    """"Left" ou "Right" -- a mao REAL do usuario. Ja corrigido pelo espelho.

    POR QUE PRECISA SER CORRIGIDO:

    Espelhamos o frame na captura (Camera.ler) para que sinalizar nao seja
    desorientador. O MediaPipe entao recebe o frame JA ESPELHADO e classifica a
    lateralidade DO QUE ELE VE -- e a sua mao direita, espelhada, tem a geometria
    de uma mao esquerda. Ele responde "Left", coerente com a imagem que recebeu,
    e errado em relacao ao mundo.

    A correcao acontece na FRONTEIRA (no unico ponto onde os dados do MediaPipe
    entram no sistema), o que mantem UMA convencao -- e verdadeira. Ver ADR-011.

    Um campo chamado `lado` que diz "Left" para a mao direita e uma armadilha
    esperando alguem -- inclusive voce, daqui a tres meses, montando o dataset.

    IMPORTANTE: trocamos o ROTULO, nao a GEOMETRIA. Os landmarks continuam vindo
    da imagem espelhada. A mesma mao real produz sempre a mesma geometria E o
    mesmo rotulo, na coleta e na inferencia -- a consistencia esta preservada.
    """

    lado_bruto: str
    """O que o MediaPipe respondeu, sem correcao. Guardado para depuracao.

    Se um dia os rotulos parecerem trocados, comparar `lado` com `lado_bruto`
    responde na hora se o problema e o espelho ou o proprio MediaPipe.
    """

    confianca_lado: float
    """Quao certo o MediaPipe esta sobre o lado. Baixo = mao ambigua no quadro."""

    landmarks: np.ndarray
    """(21, 3) float32. Coordenadas NORMALIZADAS pela imagem.

    x, y em [0, 1] -- fracao da largura/altura do frame. z e a profundidade
    relativa ao PULSO, na mesma escala de x (negativo = mais perto da camera).

    Bom para DESENHAR (multiplica por largura/altura e voce tem o pixel).
    Ruim como FEATURE: se voce anda para o lado, todos os numeros mudam,
    embora o sinal seja o mesmo. Use `world` para isso.
    """

    world: np.ndarray
    """(21, 3) float32. Coordenadas METRICAS 3D, em METROS.

    Origem no centro geometrico da mao. Nao dependem de onde a mao esta no
    frame nem de quao longe da camera ela esta: o mesmo gesto produz
    aproximadamente os mesmos numeros.

    E daqui que vao sair as features do classificador. Metade do "feature
    engineering" ja vem pronto -- invariancia a translacao e escala -- de graca,
    por usarmos um modelo pre-treinado.
    """

    @property
    def pulso(self) -> np.ndarray:
        return self.landmarks[0]


@dataclass(slots=True, frozen=True)
class ResultadoMaos:
    maos: tuple[Mao, ...]

    @property
    def vazio(self) -> bool:
        return len(self.maos) == 0

    def por_lado(self, lado: str) -> Mao | None:
        return next((m for m in self.maos if m.lado == lado), None)
