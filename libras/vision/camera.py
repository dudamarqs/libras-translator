"""Captura de video da webcam.

Esta e a camada mais baixa do sistema: tudo depois dela (MediaPipe, coleta de
dataset, inferencia) consome os frames produzidos aqui.

Duas garantias que este modulo oferece e que um `cv2.VideoCapture(0)` cru nao
oferece:

1. **A camera SEMPRE e liberada.** No Windows a webcam e um recurso exclusivo:
   um processo por vez. Se o script morrer com uma excecao antes do
   `cap.release()`, o handle fica preso e a proxima execucao falha com
   "nao consegui abrir a webcam" -- e voce vai procurar o bug no lugar errado.
   Por isso `Camera` e um context manager (`with`): a liberacao acontece mesmo
   se o corpo do `with` explodir.

2. **O frame ja sai no formato canonico do projeto** (espelhado). Ver a secao
   sobre lateralidade em `Camera.ler()`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType

import cv2
import numpy as np


class CameraError(RuntimeError):
    """A webcam nao pode ser aberta ou parou de entregar frames."""


@dataclass(slots=True, frozen=True)
class CameraConfig:
    """Configuracao da captura.

    Um dataclass -- e nao seis argumentos soltos -- porque essa configuracao vai
    precisar ser SALVA junto com o dataset. Se voce coletar dados a 640x480 e
    depois rodar a inferencia a 1280x720, os landmarks normalizados ate
    sobrevivem (eles sao relativos), mas o campo de visao muda e o MediaPipe se
    comporta diferente. Configuracao de captura e metadado do dataset.
    """

    indice: int = 0
    """Qual camera. 0 = a primeira que o SO listar (a integrada, tipicamente)."""

    largura: int = 640
    altura: int = 480
    """640x480 e proposital, nao preguica.

    O MediaPipe redimensiona a entrada internamente de qualquer jeito -- resolucao
    maior NAO melhora a deteccao de landmarks, so gasta CPU copiando pixels que
    serao jogados fora. 640x480 mantem o loop folgado a 30 FPS."""

    espelhar: bool = True
    """Ver a nota sobre lateralidade em `Camera.ler()`."""

    backend: int = cv2.CAP_MSMF
    """Media Foundation (MSMF). ESCOLHIDO POR MEDICAO, nao por intuicao.

    A primeira versao deste arquivo usava cv2.CAP_DSHOW (DirectShow), com um
    comentario confiante explicando que ele "abre a camera instantaneamente
    enquanto o MSMF demora ~2s". O comentario estava certo -- e a escolha,
    errada. Benchmark nesta maquina (640x480):

        DSHOW : 15.0 FPS  (66.7 ms/frame)  codec YUY2
        MSMF  : 29.5 FPS  (33.9 ms/frame)

    O DSHOW ficava preso em YUY2 (video NAO comprimido: ~614 KB por frame) e
    ignorava tanto o pedido de 30 FPS quanto a troca para MJPG -- o driver
    cortava a taxa pela metade para nao estourar a banda do USB.

    Abrir a camera acontece UMA VEZ; ler frames acontece 30x por segundo, para
    sempre. Trocar 2s de inicializacao por metade do FPS e um pessimo negocio.

    Em Linux/Mac use cv2.CAP_ANY. Ver ADR-007."""


class Camera:
    """Webcam como context manager.

    Uso:

        with Camera() as cam:
            for frame in cam.frames():
                cv2.imshow("preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._cap: cv2.VideoCapture | None = None

    # -- ciclo de vida -----------------------------------------------------

    def abrir(self) -> Camera:
        cfg = self.config
        cap = cv2.VideoCapture(cfg.indice, cfg.backend)

        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"nao consegui abrir a camera no indice {cfg.indice}. "
                "No Windows a webcam e exclusiva: feche Teams / Zoom / Meet / "
                "o app Camera e tente de novo."
            )

        # Pedir NAO e obter: o driver aceita o que quer. Por isso relemos os
        # valores efetivos logo abaixo, em vez de confiar no que pedimos.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.largura)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.altura)

        # LATENCIA vs FPS -- o ajuste mais importante deste arquivo.
        #
        # A camera enche um buffer interno enquanto voce processa. Se o seu
        # processamento for mais lento que a captura, o buffer ACUMULA e voce
        # passa a processar frames do passado: o FPS parece otimo, mas a imagem
        # na tela esta 300ms atras da sua mao. Para traducao em tempo real isso
        # e fatal -- o usuario faz o sinal e a resposta vem tarde.
        #
        # Buffer = 1 significa: "descarte frames antigos, me de sempre o mais
        # recente". Trocamos throughput (frames vistos) por latencia (frescor).
        # Nem todo driver respeita este flag, mas quando respeita, muda tudo.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        return self

    def fechar(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> Camera:
        return self.abrir()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Roda mesmo se o corpo do `with` levantou excecao. E este o ponto.
        self.fechar()
        cv2.destroyAllWindows()

    # -- leitura -----------------------------------------------------------

    @property
    def resolucao_efetiva(self) -> tuple[int, int]:
        """O que o driver REALMENTE entregou (nao o que pedimos)."""
        if self._cap is None:
            raise CameraError("camera nao esta aberta")
        largura = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return largura, altura

    def ler(self) -> np.ndarray:
        """Le um frame. Levanta CameraError se falhar.

        Retorna um array BGR de shape (altura, largura, 3), dtype uint8.

        LATERALIDADE -- ler com atencao:

        Espelhamos o frame AQUI, uma unica vez, e o frame espelhado passa a ser
        a UNICA realidade do sistema: e ele que vai para a tela, para o
        MediaPipe, para a coleta do dataset e para a inferencia.

        Por que espelhar: sem isso, a webcam te mostra como os outros te veem --
        voce levanta a mao direita e ela aparece do lado esquerdo da tela.
        Sinalizar assim e desorientador.

        Por que espelhar SO UMA VEZ, e antes de tudo: o MediaPipe informa se a
        mao detectada e esquerda ou direita, e isso e uma feature real (em Libras
        existe mao dominante). Num frame espelhado, ele vai chamar sua mao
        direita de "Left". Isso NAO e um problema -- desde que seja consistente.
        Se voce espelhasse apenas na exibicao e treinasse com o frame original,
        criaria um `training/serving skew` de lateralidade: o modelo aprenderia
        uma convencao e receberia a oposta em producao.

        Uma convencao, aplicada num lugar so. Ver ADR-004.
        """
        if self._cap is None:
            raise CameraError("camera nao esta aberta -- use `with Camera() as cam:`")

        # cap.read() devolve (sucesso, frame). Ignorar o `sucesso` e o erro mais
        # comum: quando a camera falha um frame -- e ela falha --, `frame` vem
        # None e a linha seguinte estoura um AttributeError incompreensivel.
        sucesso, frame = self._cap.read()
        if not sucesso or frame is None:
            raise CameraError("a camera abriu mas parou de entregar frames")

        if self.config.espelhar:
            frame = cv2.flip(frame, 1)  # 1 = horizontal

        return frame

    def frames(self) -> Iterator[np.ndarray]:
        """Gera frames indefinidamente. Interrompa com `break`."""
        while True:
            yield self.ler()
