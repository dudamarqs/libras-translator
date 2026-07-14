"""Deteccao de maos com o MediaPipe (API `tasks`).

Este modulo e a fronteira entre o mundo dos PIXELS e o mundo dos NUMEROS.
Antes dele: arrays de 921.600 bytes. Depois dele: 21 pontos 3D por mao.

Tudo que vier a seguir no projeto -- coleta de dataset, treino, inferencia --
consome a saida daqui. Por isso ela vive em `libras/` e nao dentro de um script:
o `training/collect.py` e o `backend/` precisam extrair features EXATAMENTE do
mesmo jeito, ou o modelo erra em producao sem levantar excecao (ADR-004).
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)


@contextlib.contextmanager
def _stderr_nativo_silenciado() -> Iterator[None]:
    """Silencia o stderr do codigo C++ (nao o do Python).

    O MediaPipe despeja, ao carregar o modelo, avisos como:

        W0000 ... Feedback manager requires a model with a single signature
        W0000 ... Using NORM_RECT without IMAGE_DIMENSIONS ...

    Sao inofensivos, nao ha o que corrigir do nosso lado, e escondem os nossos
    proprios prints.

    A primeira tentativa foi `os.environ["GLOG_minloglevel"] = "2"` antes do
    import. Nao funcionou -- e a razao vale aprender: o MediaPipe migrou de glog
    para o logging do `absl`, que IGNORA essa variavel. O codigo ficava la,
    parecendo resolver, sem resolver nada.

    Estes avisos nao passam pelo `logging` do Python. Sao escritos DIRETO no
    file descriptor 2 pelo C++. Nenhum `contextlib.redirect_stderr` os pega --
    ele so troca o objeto `sys.stderr` do Python, e o C++ nao sabe que ele
    existe. E preciso trocar o descritor no nivel do SISTEMA OPERACIONAL, que e
    o que `os.dup2` faz.

    Escopo: usado SO durante a abertura do modelo, onde os avisos sao emitidos.
    Nao envolvemos o loop de inferencia -- silenciar stderr permanentemente
    esconderia erros de verdade.
    """
    sys.stderr.flush()
    fd_stderr = sys.stderr.fileno()
    copia = os.dup(fd_stderr)  # guarda o stderr original
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fd_stderr)  # fd 2 passa a apontar para o buraco negro
        yield
    finally:
        sys.stderr.flush()
        os.dup2(copia, fd_stderr)  # devolve o stderr de verdade
        os.close(copia)
        os.close(devnull)


CAMINHO_MODELO_PADRAO = (
    Path(__file__).resolve().parents[2] / "models" / "mediapipe" / "hand_landmarker.task"
)

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


@dataclass(slots=True, frozen=True)
class Mao:
    """Uma mao detectada."""

    lado: str
    """"Left" ou "Right" -- COMO O MEDIAPIPE VE, num frame JA ESPELHADO.

    Como espelhamos o frame na captura (ver Camera.ler), o que o MediaPipe chama
    de "Left" e, na vida real, a sua mao DIREITA. Isso NAO e um bug e nao vamos
    "consertar": e uma convencao, e o que importa e ela ser a MESMA na coleta,
    no treino e na inferencia. Trocar os rotulos so aqui, "para ficar certo",
    e como criar duas convencoes e escolher a errada em algum lugar.
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

    E daqui que vao sair as features do classificador (Etapa 7). Repare que
    metade do trabalho de "feature engineering" ja vem pronto -- invariancia a
    translacao e a escala -- de graca, por usarmos um modelo pre-treinado.
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


class DetectorMaos:
    """Wrapper do HandLandmarker.

        with DetectorMaos() as detector, Camera() as cam:
            for frame in cam.frames():
                resultado = detector.detectar(frame)

    POR QUE ISTO E RAPIDO -- o truque que explica os numeros que voce vai medir:

    O `HandLandmarker` nao e um modelo, sao DOIS:

      1. palm detector  -- varre o frame INTEIRO procurando "onde tem mao?".
                           E o caro.
      2. landmark model -- recebe um RECORTE que ja contem a mao e devolve os
                           21 pontos. E o barato.

    Rodar os dois em todo frame seria lento demais. Entao ele roda o detector
    UMA vez, acha a mao, e nos frames seguintes usa a posicao anterior para
    recortar a regiao e rodar so o modelo barato. Ele so volta a chamar o
    detector caro quando PERDE o rastreamento.

    Isso so funciona no modo VIDEO/LIVE_STREAM, que mantem estado entre frames.
    No modo IMAGE cada frame e tratado como uma foto isolada e o detector roda
    SEMPRE -- por isso ele e varias vezes mais lento. Rode
    `scripts/bench_maos.py` para ver a diferenca com os proprios olhos.
    """

    def __init__(
        self,
        *,
        caminho_modelo: Path | None = None,
        max_maos: int = 2,
        conf_deteccao: float = 0.5,
        conf_presenca: float = 0.5,
        conf_rastreamento: float = 0.5,
        modo_video: bool = True,
    ) -> None:
        self.caminho_modelo = caminho_modelo or CAMINHO_MODELO_PADRAO
        self.max_maos = max_maos
        self.conf_deteccao = conf_deteccao
        self.conf_presenca = conf_presenca
        self.conf_rastreamento = conf_rastreamento
        self.modo_video = modo_video

        self._landmarker: HandLandmarker | None = None
        self._ultimo_ts_ms = -1

    # -- ciclo de vida -----------------------------------------------------

    def abrir(self) -> DetectorMaos:
        if not self.caminho_modelo.exists():
            raise FileNotFoundError(
                f"modelo nao encontrado em {self.caminho_modelo}\n"
                "Rode:  .venv\\Scripts\\python.exe scripts\\download_models.py"
            )

        opcoes = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.caminho_modelo)),
            running_mode=(
                VisionTaskRunningMode.VIDEO if self.modo_video else VisionTaskRunningMode.IMAGE
            ),
            num_hands=self.max_maos,
            # Age quando o DETECTOR roda: "quao certo preciso estar de que isto
            # e uma mao para comecar a rastrear".
            min_hand_detection_confidence=self.conf_deteccao,
            min_hand_presence_confidence=self.conf_presenca,
            # Age em TODO frame rastreado: "abaixo disto eu desisto do
            # rastreamento e chamo o detector caro de novo".
            min_tracking_confidence=self.conf_rastreamento,
        )

        with _stderr_nativo_silenciado():
            self._landmarker = HandLandmarker.create_from_options(opcoes)
            self._aquecer()

        return self

    def _aquecer(self) -> None:
        """Roda inferencias descartaveis num frame preto.

        A PRIMEIRA inferencia e varias VEZES mais cara que as seguintes: ela
        carrega os pesos, aloca os buffers nativos e inicializa o delegate
        XNNPACK (o motor de CPU do TensorFlow Lite).

        Isso ja envenenou uma medicao nossa: a primeira versao do
        `bench_maos.py` aqueceu de menos e o primeiro cenario mediu 55 ms
        contra 16 ms dos demais. Era cold start disfarcado de resultado.

        Aquecer AQUI DENTRO, e nao em cada script, garante que ninguem mais
        esqueca -- e faz o primeiro frame do preview ja sair no ritmo certo,
        sem aquele engasgo inicial.
        """
        preto = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.detectar(preto)

    def fechar(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()  # libera os buffers nativos (C++)
            self._landmarker = None

    def __enter__(self) -> DetectorMaos:
        return self.abrir()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.fechar()

    # -- deteccao ----------------------------------------------------------

    def detectar(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> ResultadoMaos:
        """Extrai as maos de um frame BGR (o formato que a Camera entrega)."""
        if self._landmarker is None:
            raise RuntimeError("detector nao esta aberto -- use `with DetectorMaos() as d:`")

        # BGR -> RGB. O OpenCV entrega BGR (heranca dos bitmaps do Windows nos
        # anos 2000); o MediaPipe foi treinado com RGB.
        #
        # ESQUECER ESTA LINHA E O BUG SILENCIOSO CLASSICO: nao levanta excecao,
        # nao quebra nada -- o modelo so passa a detectar a mao PIOR, porque
        # esta recebendo as cores trocadas. Voce ficaria dias culpando a
        # iluminacao da sua sala.
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self.modo_video:
            # O modo VIDEO EXIGE timestamps ESTRITAMENTE CRESCENTES. Se voce
            # repetir ou regredir um timestamp, o MediaPipe levanta excecao --
            # e a mensagem nao ajuda em nada. Por isso o detector mantem o
            # proprio contador em vez de confiar em quem chama.
            if timestamp_ms is None or timestamp_ms <= self._ultimo_ts_ms:
                timestamp_ms = self._ultimo_ts_ms + 1
            self._ultimo_ts_ms = timestamp_ms
            bruto = self._landmarker.detect_for_video(imagem, timestamp_ms)
        else:
            bruto = self._landmarker.detect(imagem)

        return self._converter(bruto)

    @staticmethod
    def _converter(bruto: HandLandmarkerResult) -> ResultadoMaos:
        """Traduz o objeto do MediaPipe para os nossos dataclasses + numpy.

        Por que nao devolver o objeto do MediaPipe direto:

        Isolamento. Se um dia trocarmos o MediaPipe (ou ele mudar de API de novo
        -- ja mudou uma vez, ver ADR-006), so ESTE metodo muda. Todo o resto do
        projeto fala com `Mao` e `ResultadoMaos`, que sao nossos.

        E, na pratica, arrays numpy sao o que o resto do pipeline quer: uma
        lista de objetos com atributos .x/.y/.z teria que ser convertida para
        array em algum lugar de qualquer jeito. Melhor num lugar so.
        """
        maos: list[Mao] = []

        for i, lms in enumerate(bruto.hand_landmarks):
            normalizados = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)

            # hand_world_landmarks vem na MESMA ordem de hand_landmarks.
            world = np.array(
                [[p.x, p.y, p.z] for p in bruto.hand_world_landmarks[i]],
                dtype=np.float32,
            )

            categoria = bruto.handedness[i][0]  # top-1 da classificacao de lado
            maos.append(
                Mao(
                    lado=categoria.category_name,
                    confianca_lado=float(categoria.score),
                    landmarks=normalizados,
                    world=world,
                )
            )

        return ResultadoMaos(maos=tuple(maos))
