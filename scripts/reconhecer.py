"""Datilologia AO VIVO com LEGENDA -- o projeto inteiro de ponta a ponta.

    .venv\\Scripts\\python.exe scripts\\reconhecer.py

Webcam -> MediaPipe (landmarks) -> features (63) -> classificador -> letras ->
montador -> corretor -> LEGENDA em portugues na tela.

Ate a Etapa 10 este script parava na terceira seta: ele mostrava a soletracao
crua ("CAURO") e a camada de texto so rodava numa demo sem camera. A Etapa 11
fechou o circuito -- e o resultado e o objetivo declarado do projeto: voce
sinaliza, aparece legenda.

Teclas:
    q / ESC     sair
    ESPACO      limpar tudo
    BACKSPACE   apagar a ultima letra
    ENTER       fechar a frase (ela vira contexto e a legenda recomeca)

O QUE OBSERVAR:
  - A linha AMARELA e o que o reconhecedor viu (crua, com os erros de R/U e
    T/F que medimos nas Etapas 8-9). A linha VERDE e a legenda corrigida.
    A distancia entre as duas e exatamente o valor da camada de texto.
  - "corrigindo..." aparece enquanto a correcao viaja. Repare que o VIDEO NAO
    TRAVA nesse tempo: a correcao roda em outra thread (ver libras/texto/
    legenda.py). Se travasse, seria bug -- e o motivo daquele modulo existir.
  - Qual corretor esta ativo aparece no console ao iniciar. Sem chave de API o
    sistema roda de graca no dicionario offline (ADR-020).
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from libras.models.classificador import Classificador
from libras.texto import (
    CorretorLLM,
    CorretorLocal,
    LegendaAoVivo,
    criar_corretor,
    pyspellchecker_disponivel,
)
from libras.vision.camera import Camera, CameraConfig, CameraError
from libras.vision.features import vetor_uma_mao
from libras.vision.hands import DetectorMaos
from libras.vision.overlay import (
    AMARELO,
    CINZA,
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

# Quanto tempo a mao precisa ficar FORA do quadro para fechar uma palavra.
#
# Por que nao fechar na primeira pausa: para soletrar "CARRO" voce precisa
# registrar dois R seguidos, e a unica forma de fazer isso e tirar a mao e
# voltar (senao o debounce trata como a mesma letra sustentada). Se qualquer
# pausa fechasse a palavra, "CARRO" viraria "CAR RO".
#   pausa CURTA  -> so libera a letra repetida
#   pausa LONGA  -> fim de palavra (espaco)
# O corretor ainda e uma segunda linha de defesa contra o espaco errado, mas
# alimenta-lo com um texto melhor sempre da uma legenda melhor.
PAUSA_FIM_DE_PALAVRA = 1.2  # segundos

TECLA_ENTER = 13
TECLA_BACKSPACE = 8


def _letra_grande(frame, letra: str, cor) -> None:  # noqa: ANN001
    """Desenha a letra prevista, grande, no canto direito."""
    h, w = frame.shape[:2]
    texto(frame, letra, (w - 120, h // 2), escala=5.0, cor=cor, espessura=6)


def _desenhar_legenda(frame, legenda: LegendaAoVivo) -> None:  # noqa: ANN001
    """Duas linhas na base: a soletracao crua e a legenda corrigida."""
    h = frame.shape[0]

    bruto = legenda.texto_bruto
    sufixo = "  corrigindo..." if legenda.corrigindo else ""
    texto(frame, f"{bruto}{sufixo}", (10, h - 55), escala=0.7, cor=AMARELO, espessura=2)

    # A legenda so aparece depois da primeira correcao. Enquanto nao ha nada
    # corrigido, `legenda.legenda` e "" e a linha fica limpa -- o bruto acima
    # ja mostra o progresso, entao nao ha momento em que a tela pareca morta.
    if legenda.legenda:
        texto(frame, f"> {legenda.legenda}", (10, h - 20), escala=1.0, cor=VERDE, espessura=2)

    if legenda.erro:
        texto(frame, f"corretor: {legenda.erro[:60]}", (10, h - 85), escala=0.5, cor=CINZA)


def _criar_legenda() -> LegendaAoVivo:
    """Escolhe o corretor disponivel e anuncia qual e (custo importa)."""
    principal = criar_corretor()
    reserva = None
    if isinstance(principal, CorretorLLM) and pyspellchecker_disponivel():
        # O offline vira rede de seguranca: se a chamada ao Claude falhar
        # (rede, credito), a legenda continua saindo em vez de sumir.
        reserva = CorretorLocal()
        print("Corretor: Claude (pago) com fallback offline.")
    elif isinstance(principal, CorretorLocal):
        print("Corretor: dicionario offline (gratis, sem contexto).")
    else:
        print("Corretor: nenhum -- a legenda sera a soletracao crua.")
    return LegendaAoVivo(principal, fallback=reserva)


def main() -> int:
    try:
        classificador = Classificador.carregar()
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return 1
    conhecidas = " ".join(classificador.classes)
    print(f"Modelo carregado. Letras conhecidas: {conhecidas}")

    config = CameraConfig()
    ultima_letra: str | None = None
    contagem = 0
    letra_ja_digitada: str | None = None
    vetor_anterior = None  # para medir o movimento entre frames
    sem_mao_desde: float | None = None  # inicio da pausa atual

    try:
        with (
            Camera(config) as cam,
            # Datilologia = 1 mao (max_maos=1). entrada_espelhada casada com a
            # camera, como em todo o projeto -- senao os landmarks viriam de uma
            # geometria espelhada da que o modelo treinou.
            DetectorMaos(max_maos=1, entrada_espelhada=config.espelhar) as detector,
            _criar_legenda() as legenda,
        ):
            print("q/ESC = sair | ESPACO = limpar | BACKSPACE = apagar | ENTER = nova frase\n")

            while True:
                frame = cam.ler()
                resultado = detector.detectar(frame)
                desenhar_maos(frame, resultado.maos)

                if resultado.vazio:
                    # Sem mao: nao ha o que classificar. Zeramos o contador para
                    # que tirar a mao "encerre" a letra atual -- e o gesto natural
                    # de separar uma letra da proxima (inclusive letra repetida).
                    ultima_letra = None
                    contagem = 0
                    letra_ja_digitada = None
                    vetor_anterior = None

                    agora = time.monotonic()
                    if sem_mao_desde is None:
                        sem_mao_desde = agora
                    elif agora - sem_mao_desde >= PAUSA_FIM_DE_PALAVRA:
                        # Pausa longa: fecha a palavra e pede a correcao. O
                        # metodo ignora repeticoes, entao chamar todo frame daqui
                        # em diante nao gasta nada.
                        legenda.fim_de_palavra()

                    painel(frame, [f"mostre a mao ({conhecidas})"], cor=AMARELO)
                else:
                    sem_mao_desde = None
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
                        legenda.letra(pred.letra)
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

                _desenhar_legenda(frame, legenda)

                cv2.imshow("Libras - datilologia ao vivo", frame)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), 27):
                    break
                if tecla == ord(" "):
                    legenda.limpar()
                elif tecla == TECLA_BACKSPACE:
                    legenda.apagar()
                elif tecla == TECLA_ENTER:
                    legenda.nova_frase()

            # Antes de sair, deixa a ultima correcao chegar: aqui bloquear e o
            # certo (nao ha mais frame para desenhar).
            legenda.fim_de_palavra()
            legenda.aguardar(timeout=10.0)
            final = legenda.texto

    except CameraError as exc:
        print(f"\nERRO DE CAMERA: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\ninterrompido")
        return 0

    print(f"\nlegenda final: {final!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
