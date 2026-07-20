"""Reconhecimento de datilologia AO VIVO -- o loop fechado.

    .venv\\Scripts\\python.exe scripts\\reconhecer.py

Webcam -> MediaPipe (landmarks) -> features (63) -> classificador -> letra na
tela. E a primeira vez que o projeto inteiro roda de ponta a ponta: da sua mao
na frente da camera ao texto, em tempo real.

Teclas:  q / ESC = sair   |   especo = limpar o texto acumulado

O QUE OBSERVAR (isto vira a Etapa 9):
  - A barra de confianca. O modelo SEMPRE aponta uma letra, mesmo quando voce
    nao faz sinal nenhum. So aceitamos a predicao quando ela passa do LIMIAR --
    e por isso o texto so cresce quando a barra fica verde.
  - Tente A, B, C, L, O (as unicas que este modelo conhece). Qualquer outra
    coisa ele vai forcar para a letra mais parecida, com confianca baixa.
"""

from __future__ import annotations

import cv2
import numpy as np

from libras.models.classificador import Classificador
from libras.vision.camera import Camera, CameraConfig, CameraError
from libras.vision.features import vetor_uma_mao
from libras.vision.hands import DetectorMaos
from libras.vision.overlay import (
    AMARELO,
    VERDE,
    VERMELHO,
    barra_confianca,
    desenhar_maos,
    painel,
    texto,
)

# So aceitamos a letra acima disto. Escolhido a olho por ora; a Etapa 9 vai
# calibrar isto com dados (curva de precisao x cobertura). Ver barra_confianca.
LIMIAR_ACEITE = 0.80

# Quantos frames seguidos a mesma letra precisa vencer para ser "digitada".
# Sem isto, um frame ruim no meio de um sinal cravaria uma letra errada. E o
# equivalente ao "debounce" de um botao fisico.
FRAMES_PARA_CONFIRMAR = 6

# Trava de IMOBILIDADE. Uma letra e um sinal SUSTENTADO; coçar a cabeca ou
# trocar de letra e MOVIMENTO. So classificamos quando a FORMA da mao esta
# parada. Medimos a mudanca do vetor de features entre frames (invariante a
# posicao: mover a mao parada pela tela nao conta como movimento -- so mudar a
# forma dos dedos conta). Acima disto, a mao esta "em transito".
#   O numero aparece no HUD ("mov: X.XX") justamente para voce calibrar a olho:
#   se letras nao registram, aumente; se lixo passa, diminua.
MOV_MAX = 0.18


def _letra_grande(frame, letra: str, cor) -> None:  # noqa: ANN001
    """Desenha a letra prevista, grande, no canto direito."""
    h, w = frame.shape[:2]
    texto(frame, letra, (w - 120, h // 2), escala=5.0, cor=cor, espessura=6)


def main() -> int:
    try:
        classificador = Classificador.carregar()
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return 1
    print(f"Modelo carregado. Letras conhecidas: {classificador.classes}")

    config = CameraConfig()
    texto_acumulado = ""
    ultima_letra: str | None = None
    contagem = 0
    letra_ja_digitada: str | None = None
    vetor_anterior = None  # para medir o movimento entre frames

    try:
        with (
            Camera(config) as cam,
            # Datilologia = 1 mao (max_maos=1). entrada_espelhada casada com a
            # camera, como em todo o projeto -- senao os landmarks viriam de uma
            # geometria espelhada da que o modelo treinou.
            DetectorMaos(max_maos=1, entrada_espelhada=config.espelhar) as detector,
        ):
            print("q/ESC = sair | espaco = limpar texto\n")

            while True:
                frame = cam.ler()
                resultado = detector.detectar(frame)
                desenhar_maos(frame, resultado.maos)

                if resultado.vazio:
                    # Sem mao: nao ha o que classificar. Zeramos o contador para
                    # que tirar a mao "encerre" a letra atual -- e o gesto natural
                    # de separar uma letra da proxima.
                    ultima_letra = None
                    contagem = 0
                    letra_ja_digitada = None
                    vetor_anterior = None
                    painel(frame, ["mostre a mao (A B C L O)"], cor=AMARELO)
                else:
                    vetor = vetor_uma_mao(resultado)
                    pred = classificador.prever(vetor)

                    # Quanto a FORMA da mao mudou desde o frame anterior.
                    movimento = (
                        float(np.abs(vetor - vetor_anterior).mean())
                        if vetor_anterior is not None
                        else 0.0
                    )
                    vetor_anterior = vetor

                    parado = movimento <= MOV_MAX
                    # Uma letra so vale se: (1) o modelo RECONHECE a forma
                    # (nao e novidade), (2) esta confiante e (3) a mao esta
                    # PARADA. As tres barreiras atacam sintomas diferentes:
                    # novidade mata o "coçar virou C"; imobilidade mata o
                    # "gravou no meio do movimento".
                    if pred.desconhecido:
                        estado, cor, letra_mostrar = "desconhecido", VERMELHO, "?"
                    elif not parado:
                        estado, cor, letra_mostrar = "movendo...", AMARELO, pred.letra
                    elif pred.confianca < LIMIAR_ACEITE:
                        estado, cor, letra_mostrar = "incerto", AMARELO, pred.letra
                    else:
                        estado, cor, letra_mostrar = "ok", VERDE, pred.letra

                    aceito = estado == "ok"

                    # Estabilidade: conta frames seguidos da MESMA letra aceita.
                    # So digita quando estabiliza, e so uma vez por gesto
                    # (letra_ja_digitada evita repetir com a mao parada no sinal).
                    if aceito and pred.letra == ultima_letra:
                        contagem += 1
                    else:
                        contagem = 1 if aceito else 0
                    ultima_letra = pred.letra if aceito else None

                    if contagem >= FRAMES_PARA_CONFIRMAR and pred.letra != letra_ja_digitada:
                        texto_acumulado += pred.letra
                        letra_ja_digitada = pred.letra

                    _letra_grande(frame, letra_mostrar, cor)
                    barra_confianca(frame, pred.confianca, canto=(10, 40), limiar=LIMIAR_ACEITE)
                    painel(
                        frame,
                        [
                            f"{estado}   letra: {pred.letra}",
                            f"mov: {movimento:.2f}   dist: {pred.distancia:.1f}",
                        ],
                        cor=cor,
                    )

                # O texto que voce vem soletrando, na base da tela.
                h = frame.shape[0]
                texto(
                    frame, f"> {texto_acumulado}", (10, h - 20),
                    escala=1.0, cor=VERDE, espessura=2,
                )

                cv2.imshow("Libras - datilologia ao vivo", frame)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), 27):
                    break
                if tecla == ord(" "):
                    texto_acumulado = ""

    except CameraError as exc:
        print(f"\nERRO DE CAMERA: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\ninterrompido")

    print(f"\ntexto final: {texto_acumulado!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
