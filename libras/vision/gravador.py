"""Grava o proprio frame do demo em GIF -- sem gravador de tela.

Por que nao usar um gravador de tela: o que a janela mostra JA esta na memoria,
pronto e desenhado. Capturar de fora significa recomprimir pixels que voce ja
tem, junto com cursor, barra de titulo e o que estiver atras da janela. Gravando
daqui, o GIF sai com exatamente o que o `overlay` desenhou, e nada mais.

Duas decisoes valem explicacao, porque as duas alternativas obvias dao errado:

1. AMOSTRAGEM PELO RELOGIO, nao "guarde 1 a cada N frames". O loop do demo nao
   roda em FPS constante -- o MediaPipe as vezes gasta 60 ms num frame. Contando
   frames, esses trechos viram camera lenta no GIF. Contando tempo, o GIF sai no
   ritmo real, que e o unico ritmo honesto para mostrar um sistema "ao vivo".

2. UMA PALETA PARA O GIF INTEIRO, TIRADA DE UMA AMOSTRA. O GIF guarda no maximo
   256 cores por quadro. Se cada quadro escolher as suas, a paleta muda a cada
   quadro e o video inteiro "ferve" (o fundo troca de tom sozinho). Entao a
   paleta e uma so -- mas NAO a do primeiro quadro: o primeiro quadro e o pior
   candidato possivel, porque e o instante em que a camera ainda esta ajustando
   exposicao e a mao normalmente nem esta em cena. Uma paleta tirada dali
   descolore o resto do clipe (e, no limite, colapsa quadros distintos no mesmo
   quadro). Guardamos as primeiras QUADROS_PARA_PALETA capturas, montamos a
   paleta a partir de todas elas juntas e so entao seguimos em streaming. E a
   ideia do `palettegen` do ffmpeg em miniatura, com memoria limitada.
   Efeito colateral bom: quadros que compartilham paleta permitem ao Pillow
   gravar so o retangulo que MUDOU em vez do quadro inteiro.

Uso:
    grav = GravadorGif(Path("docs/demo.gif"))
    grav.alternar()             # comeca a gravar
    grav.capturar(frame)        # a cada frame do loop; ele decide se guarda
    caminho = grav.salvar()     # escreve o arquivo
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from libras.vision.overlay import AMARELO, AZUL, BRANCO, CINZA, PRETO, VERDE, VERMELHO

# GIF nao e video: cada quadro e uma imagem inteira, entao o arquivo cresce
# rapido. Estes tres numeros sao o orcamento de tamanho, e todos custam algo:
#   FPS_PADRAO   -- 10 q/s ja le como movimento; 30 triplicaria o arquivo.
#   LARGURA_PADRAO -- README do GitHub exibe a ~800 px; 560 e nitido la e pesa 3x
#                     menos que 1280.
#   CORES_PADRAO -- 64 cores bastam para pele, fundo e o HUD (que e chapado).
FPS_PADRAO = 10
LARGURA_PADRAO = 560
CORES_PADRAO = 64

# Trava de seguranca: 20 s a 10 q/s = 200 quadros na RAM. Um GIF de README que
# passa disso ninguem assiste ate o fim -- e o teto existe para voce nao
# descobrir o problema depois de gravar tres minutos.
SEGUNDOS_MAX_PADRAO = 20.0

# Quantas capturas entram na amostra que decide a paleta. Oito quadros a 10 q/s
# cobrem ~0,8 s: tempo de a camera estabilizar e de a mao entrar em cena, sem
# segurar RGB demais na memoria enquanto isso.
QUADROS_PARA_PALETA = 8

# As cores do HUD entram na paleta RESERVADAS, fora da disputa.
#
# Por que: o quantizador escolhe cores por FREQUENCIA, e o HUD ocupa uns poucos
# milhares de pixels contra ~235 mil de rosto e parede. Ele perde a votacao
# sempre, e suas cores sao absorvidas pelo tom de pele mais proximo. Medido na
# primeira gravacao real: o amarelo (250,200,60) da legenda saiu como
# (180,128,115) -- um marrom -- e o verde da letra grande perdeu TODA a
# saturacao. Isso nao e perda de qualidade, e perda de INFORMACAO: no demo a cor
# E o significado (verde = aceito, amarelo = incerto, vermelho = desconhecido).
# Reservando as sete cores, elas sobrevivem exatas e custam 7 das 64 vagas.
CORES_DO_HUD = tuple(
    tuple(reversed(cor))  # o overlay fala BGR (OpenCV); aqui e tudo RGB
    for cor in (BRANCO, PRETO, VERDE, AMARELO, VERMELHO, AZUL, CINZA)
)

# O maior inimigo do tamanho do GIF nao e o movimento: e o RUIDO DO SENSOR.
# Cada pixel da parede treme 2-3 niveis por quadro, e isso destroi as duas
# compressoes que o formato tem -- o LZW nao acha sequencias repetidas, e o
# Pillow, que grava so o retangulo que mudou entre quadros, e obrigado a gravar
# o quadro inteiro porque "tudo mudou um pouquinho".
#   Medido num clipe SIMULADO de 15 s (fundo estatico + ruido gaussiano + um
#   bloco se movendo, 560 px, 10 q/s): 12,89 MB sem tratar, 0,32 MB tratando --
#   40x. Numa gravacao real espere mais que isso: a camera treme na mao e a luz
#   oscila, entao o fundo nunca fica tao parado quanto no simulado.
# O tratamento e congelar o pixel que mudou MENOS que isto (0-255 por canal):
# a parede vira de fato estatica, e a mao, que muda muito mais, passa intacta.
TOLERANCIA_RUIDO = 10


class GravadorGif:
    """Acumula quadros do demo e escreve um GIF animado ao final."""

    def __init__(
        self,
        caminho: Path,
        *,
        fps: int = FPS_PADRAO,
        largura: int = LARGURA_PADRAO,
        cores: int = CORES_PADRAO,
        cores_reservadas: tuple[tuple[int, int, int], ...] = CORES_DO_HUD,
        tolerancia_ruido: int = TOLERANCIA_RUIDO,
        segundos_max: float = SEGUNDOS_MAX_PADRAO,
        relogio=time.monotonic,  # noqa: ANN001 -- injetado para o teste nao dormir
    ) -> None:
        if fps <= 0:
            raise ValueError("fps precisa ser positivo")
        self.caminho = Path(caminho)
        self.fps = fps
        self.largura = largura
        self.cores = cores
        self.cores_reservadas = cores_reservadas
        self.tolerancia_ruido = tolerancia_ruido
        self.segundos_max = segundos_max
        self._relogio = relogio

        self._intervalo = 1.0 / fps
        self._quadros: list[Image.Image] = []  # ja na paleta final
        self._amostra: list[Image.Image] = []  # RGB, esperando a paleta existir
        self._paleta: Image.Image | None = None
        self._anterior: np.ndarray | None = None  # RGB ja estabilizado
        self._ativo = False
        self._proximo_em = 0.0
        self._inicio = 0.0

    # -- estado ------------------------------------------------------------
    @property
    def ativo(self) -> bool:
        return self._ativo

    @property
    def n_quadros(self) -> int:
        return len(self._quadros) + len(self._amostra)

    @property
    def segundos(self) -> float:
        """Duracao que o GIF tera -- derivada dos quadros, nao do relogio.

        Sao coisas diferentes: se voce gravou 8 s e o loop entregou quadros de
        menos, o GIF dura menos. O numero que interessa e o do arquivo.
        """
        return self.n_quadros / self.fps

    def alternar(self) -> bool:
        """Liga/desliga a gravacao. Devolve o novo estado."""
        self._ativo = not self._ativo
        if self._ativo:
            agora = self._relogio()
            self._inicio = agora
            # Sem isto, o primeiro quadro depois de religar seria descartado por
            # causa do intervalo do trecho anterior.
            self._proximo_em = agora
        return self._ativo

    # -- captura -----------------------------------------------------------
    def capturar(self, frame: np.ndarray) -> bool:
        """Guarda o frame se ja passou o intervalo. Devolve se guardou.

        Recebe o frame no formato do OpenCV (BGR, in-place decorado pelo
        overlay) e guarda uma COPIA convertida -- o demo continua desenhando
        sobre o array original no proximo frame.
        """
        if not self._ativo:
            return False

        agora = self._relogio()
        if agora - self._inicio >= self.segundos_max:
            self._ativo = False
            return False
        if agora < self._proximo_em:
            return False

        # Ancora no horario ESPERADO, nao em `agora`: somar a partir do atraso
        # de cada quadro faz o erro se acumular e o GIF terminar mais lento que
        # a realidade.
        self._proximo_em += self._intervalo
        if self._proximo_em < agora:  # atraso grande (ex: correcao travou o loop)
            self._proximo_em = agora + self._intervalo

        imagem = self._reduzir(frame)
        if self._paleta is not None:
            self._quadros.append(self._na_paleta(imagem))
        else:
            self._amostra.append(imagem)
            if len(self._amostra) >= QUADROS_PARA_PALETA:
                self._fixar_paleta()
        return True

    def _reduzir(self, frame: np.ndarray) -> Image.Image:
        """BGR do OpenCV -> imagem RGB do Pillow, no tamanho de saida."""
        altura, largura = frame.shape[:2]
        if largura > self.largura:
            nova_altura = round(altura * self.largura / largura)
            # INTER_AREA e o filtro certo para DIMINUIR: faz media da regiao, em
            # vez de amostrar um pixel e criar serrilhado no HUD.
            frame = cv2.resize(frame, (self.largura, nova_altura), interpolation=cv2.INTER_AREA)
        return Image.fromarray(self._estabilizar(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    def _estabilizar(self, rgb: np.ndarray) -> np.ndarray:
        """Repete o pixel anterior onde a mudanca cabe dentro do ruido.

        Compara pelo canal que mais mudou (nao pela media): um pixel que mudou
        muito so no vermelho mudou de verdade, e a media esconderia isso.
        """
        if self.tolerancia_ruido <= 0:
            return rgb
        if self._anterior is None or self._anterior.shape != rgb.shape:
            self._anterior = rgb.copy()
            return rgb

        # abs() em uint8 daria a volta no zero (200-210 viraria 246). int16 antes.
        variacao = np.abs(rgb.astype(np.int16) - self._anterior.astype(np.int16))
        parado = variacao.max(axis=2) < self.tolerancia_ruido

        saida = rgb.copy()
        saida[parado] = self._anterior[parado]
        # Guardamos o resultado JA estabilizado: comparar sempre contra o quadro
        # cru deixaria o ruido reentrar por acumulo.
        self._anterior = saida
        return saida

    def _na_paleta(self, imagem: Image.Image) -> Image.Image:
        # dither=NONE de proposito: o dithering espalha ruido para simular cores
        # que a paleta nao tem, e ruido e justamente o que o GIF nao consegue
        # comprimir. Com o HUD chapado e so a pele em gradiente, o ganho visual
        # nao paga o tamanho.
        return imagem.quantize(palette=self._paleta, dither=Image.Dither.NONE)

    def _fixar_paleta(self) -> None:
        """Escolhe a paleta definitiva olhando a amostra inteira de uma vez.

        Empilhar as amostras numa imagem so nao e truque: o quantizador do
        Pillow decide as cores pela FREQUENCIA com que aparecem, entao dar a ele
        varios quadros de uma vez e literalmente pedir "as N cores mais comuns
        do clipe" em vez de "as N cores mais comuns deste instante".
        """
        if not self._amostra:
            return
        largura = self._amostra[0].width
        altura = sum(im.height for im in self._amostra)
        montagem = Image.new("RGB", (largura, altura))
        y = 0
        for im in self._amostra:
            montagem.paste(im, (0, y))
            y += im.height

        reservadas = self.cores_reservadas
        # As vagas reservadas saem do orcamento: a cena fica com o resto.
        vagas_cena = max(2, self.cores - len(reservadas))
        base = montagem.quantize(colors=vagas_cena)

        tabela = base.getpalette()[: vagas_cena * 3]
        for cor in reservadas:
            tabela.extend(cor)
        paleta = Image.new("P", (1, 1))
        paleta.putpalette(tabela)
        self._paleta = paleta

        self._quadros.extend(self._na_paleta(im) for im in self._amostra)
        self._amostra.clear()

    # -- saida -------------------------------------------------------------
    def salvar(self) -> Path | None:
        """Escreve o GIF. Devolve o caminho, ou None se nao ha quadros."""
        # Gravacao curta (menos que a amostra): a paleta sai do que houver.
        self._fixar_paleta()
        if not self._quadros:
            return None

        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self._quadros[0].save(
            self.caminho,
            save_all=True,
            append_images=self._quadros[1:],
            duration=round(1000 / self.fps),
            loop=0,  # 0 = repete para sempre; e o que se espera de um GIF de README
            optimize=True,
            disposal=1,  # nao apaga o quadro anterior -- e o que permite gravar so o delta
        )
        return self.caminho
