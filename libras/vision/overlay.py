"""Desenho de informacao sobre o frame (HUD).

Separado da captura de proposito. A `Camera` produz frames; o `overlay` os
decora. Misturar as duas coisas parece pratico no comeco e depois te obriga a
arrancar o desenho de dentro da captura quando o backend passar a servir frames
para o navegador -- onde quem desenha e o React, nao o OpenCV.

Convencao deste modulo: TODA funcao aqui MODIFICA o frame recebido, in-place, e
o devolve por conveniencia. E deliberado: copiar um array de 640x480x3 a cada
frame, 30 vezes por segundo, e desperdicio puro. Mas isso e uma faca -- se voce
precisar do frame limpo depois de desenhar, faca `frame.copy()` ANTES.
"""

from __future__ import annotations

import cv2
import numpy as np

from libras.vision.hands import CONEXOES, Mao

# BGR, nao RGB -- estamos no mundo do OpenCV.
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
VERDE = (80, 220, 100)
AMARELO = (60, 200, 250)
VERMELHO = (70, 70, 240)
AZUL = (240, 160, 60)
CINZA = (150, 150, 150)

_FONTE = cv2.FONT_HERSHEY_SIMPLEX

# So para EXIBIR. Internamente o projeto fala "Left"/"Right" (o vocabulario do
# MediaPipe), e a traducao acontece na borda -- a interface. Traduzir os rotulos
# la dentro obrigaria todo o codigo a lidar com dois idiomas.
LADO_PT: dict[str, str] = {"Left": "Esquerda", "Right": "Direita"}


def texto(
    frame: np.ndarray,
    conteudo: str,
    posicao: tuple[int, int],
    *,
    escala: float = 0.6,
    cor: tuple[int, int, int] = BRANCO,
    espessura: int = 1,
) -> np.ndarray:
    """Escreve texto legivel sobre QUALQUER fundo.

    O truque: desenhamos o texto duas vezes -- primeiro grosso e preto, depois
    fino e colorido por cima. Isso cria um contorno.

    Sem o contorno, texto branco some numa parede branca e texto preto some numa
    sombra. Como o fundo e a sua sala (que voce nao controla), o contorno nao e
    firula: e a unica forma de o HUD ser legivel sempre.
    """
    cv2.putText(frame, conteudo, posicao, _FONTE, escala, PRETO, espessura + 2, cv2.LINE_AA)
    cv2.putText(frame, conteudo, posicao, _FONTE, escala, cor, espessura, cv2.LINE_AA)
    return frame


def painel(
    frame: np.ndarray,
    linhas: list[str],
    *,
    canto: tuple[int, int] = (10, 24),
    escala: float = 0.55,
    cor: tuple[int, int, int] = BRANCO,
) -> np.ndarray:
    """Empilha varias linhas de texto a partir de um canto."""
    x, y = canto
    altura_linha = int(28 * escala / 0.55)
    for i, linha in enumerate(linhas):
        texto(frame, linha, (x, y + i * altura_linha), escala=escala, cor=cor)
    return frame


def barra_confianca(
    frame: np.ndarray,
    valor: float,
    *,
    canto: tuple[int, int] = (10, 10),
    tamanho: tuple[int, int] = (220, 14),
    limiar: float = 0.7,
) -> np.ndarray:
    """Barra horizontal de 0 a 1 para a confianca da predicao.

    Por que a confianca aparece na tela desde o primeiro dia, e nao "depois":

    Um classificador com softmax SEMPRE devolve uma classe -- inclusive quando
    voce nao esta fazendo sinal nenhum, ou esta fazendo um sinal que ele nunca
    viu. Ele nao tem como dizer "nao sei"; ele distribui probabilidade entre as
    classes que conhece e escolhe a maior. Se voce so mostrar o rotulo, o
    sistema parece funcionar e mente com confianca.

    A barra torna esse comportamento VISIVEL: voce ve, com os proprios olhos, o
    modelo gritando "A!" com 34% de certeza enquanto voce coca a cabeca. Isso
    vai guiar toda a Etapa 9 (avaliacao) e a decisao do limiar de aceitacao.

    A cor muda no limiar -- verde quando o sistema confia o bastante para
    aceitar a predicao, amarelo/vermelho quando nao.
    """
    valor = float(np.clip(valor, 0.0, 1.0))
    x, y = canto
    largura, altura = tamanho

    # Fundo (o "trilho" da barra).
    cv2.rectangle(frame, (x, y), (x + largura, y + altura), PRETO, -1)
    cv2.rectangle(frame, (x, y), (x + largura, y + altura), CINZA, 1)

    if valor >= limiar:
        cor = VERDE
    elif valor >= limiar * 0.6:
        cor = AMARELO
    else:
        cor = VERMELHO

    preenchido = int(largura * valor)
    if preenchido > 0:
        cv2.rectangle(frame, (x, y), (x + preenchido, y + altura), cor, -1)

    # A marca do limiar: onde fica a linha entre "aceito" e "descarto".
    marca = x + int(largura * limiar)
    cv2.line(frame, (marca, y - 2), (marca, y + altura + 2), BRANCO, 1)

    texto(frame, f"{valor:.0%}", (x + largura + 8, y + altura), escala=0.5, cor=cor)
    return frame


def desenhar_mao(frame: np.ndarray, mao: Mao, *, mostrar_indices: bool = False) -> np.ndarray:
    """Desenha o esqueleto de 21 pontos de uma mao.

    Usa `mao.landmarks` (normalizado pela imagem, 0..1) -- e NAO `mao.world`.
    Este e o unico lugar do projeto onde os landmarks normalizados sao os
    corretos: eles sao relativos ao FRAME, entao basta multiplicar pela largura
    e pela altura para virar pixel.

    O `world` (metrico, em metros, com origem no centro da mao) nao tem relacao
    nenhuma com a posicao na tela -- desenhar com ele produziria uma mao
    grudada no canto superior esquerdo. Cada espaco tem o seu uso.
    """
    altura, largura = frame.shape[:2]

    # (21, 3) normalizado -> (21, 2) em pixels.
    pontos = (mao.landmarks[:, :2] * np.array([largura, altura], dtype=np.float32)).astype(int)

    cor = VERDE if mao.lado == "Right" else AZUL

    for inicio, fim in CONEXOES:
        cv2.line(frame, tuple(pontos[inicio]), tuple(pontos[fim]), cor, 2, cv2.LINE_AA)

    for i, (px, py) in enumerate(pontos):
        # As pontas dos dedos (4, 8, 12, 16, 20) sao os pontos que mais
        # distinguem um sinal do outro -- destacamos para voce ver isso.
        e_ponta = i in (4, 8, 12, 16, 20)
        cv2.circle(frame, (px, py), 5 if e_ponta else 3, BRANCO, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 5 if e_ponta else 3, PRETO, 1, cv2.LINE_AA)

        if mostrar_indices:
            texto(frame, str(i), (px + 6, py - 6), escala=0.35, cor=AMARELO)

    # Rotulo do lado, ancorado no pulso.
    # `mao.lado` ja e a mao REAL do usuario (corrigida pelo espelho em
    # hands._lado_real). Aqui so traduzimos para portugues, para exibir.
    px, py = pontos[0]
    texto(
        frame,
        f"{LADO_PT.get(mao.lado, mao.lado)} {mao.confianca_lado:.0%}",
        (px - 20, py + 24),
        escala=0.5,
        cor=cor,
    )
    return frame


def desenhar_maos(
    frame: np.ndarray, maos: tuple[Mao, ...], *, mostrar_indices: bool = False
) -> np.ndarray:
    for mao in maos:
        desenhar_mao(frame, mao, mostrar_indices=mostrar_indices)
    return frame
