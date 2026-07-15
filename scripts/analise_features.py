"""A normalizacao das features realmente ajuda? Vamos MEDIR.

    .venv\\Scripts\\python.exe scripts\\analise_features.py

--------------------------------------------------------------------------
O QUE ESTE SCRIPT MEDE, E POR QUE ESSA E A METRICA CERTA
--------------------------------------------------------------------------

E tentador medir "invariancia": faco o mesmo gesto andando pelo quadro, e vejo
qual representacao muda menos. Mas essa metrica sozinha PREMIA O ERRADO -- uma
representacao que devolvesse sempre zero seria perfeitamente invariante e
completamente inutil.

O que um classificador precisa nao e "features estaveis". E features em que:

    gestos IGUAIS ficam PERTO      (espalhamento intra-classe pequeno)
    gestos DIFERENTES ficam LONGE  (separacao inter-classe grande)

Entao medimos a RAZAO entre as duas:

    razao = separacao_entre_gestos / espalhamento_dentro_do_gesto

Quanto MAIOR, mais facil e a classificacao. Essa razao e, essencialmente, o
problema que o modelo vai ter que resolver -- estamos medindo a dificuldade da
tarefa ANTES de treinar qualquer coisa. Se a razao for alta, ate um
RandomForest de 5 linhas acerta. Se for ~1, nem uma rede gigante salva.

(Quem conhece estatistica vai reconhecer: e o mesmo espirito do criterio de
Fisher / analise discriminante. Nao inventamos nada -- e o jeito classico de
perguntar "esta tarefa e separavel?".)

Comparamos tres representacoes dos MESMOS frames:

  A) landmarks normalizados pela IMAGEM  -- o que quase todo tutorial usa
  B) world landmarks CRUS                -- em metros, sem tratamento
  C) as nossas features                  -- world recentrado no pulso + escala
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from libras.vision.camera import Camera, CameraConfig
from libras.vision.features import normalizar_mao
from libras.vision.hands import DetectorMaos
from libras.vision.overlay import AMARELO, VERDE, painel

SEGUNDOS_POR_GESTO = 12
MIN_AMOSTRAS = 40


def coletar(rotulo: str, instrucao: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Coleta (landmarks_imagem, world) enquanto voce faz UM gesto se movendo."""
    config = CameraConfig()
    amostras: list[tuple[np.ndarray, np.ndarray]] = []

    with (
        Camera(config) as cam,
        DetectorMaos(max_maos=1, entrada_espelhada=config.espelhar) as detector,
    ):
        fim = time.perf_counter() + SEGUNDOS_POR_GESTO
        while (restante := fim - time.perf_counter()) > 0:
            frame = cam.ler()
            resultado = detector.detectar(frame)

            if not resultado.vazio:
                mao = resultado.maos[0]
                amostras.append((mao.landmarks.copy(), mao.world.copy()))

            painel(
                frame,
                [
                    f"GESTO: {rotulo}",
                    instrucao,
                    f"{restante:.0f}s   amostras: {len(amostras)}",
                ],
                cor=VERDE if not resultado.vazio else AMARELO,
            )
            cv2.imshow("Analise de features", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cv2.destroyAllWindows()
    return amostras


def estatisticas(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Espalhamento dentro de cada gesto, separacao entre eles, e a razao.

    `a` e `b` sao matrizes (n_amostras, n_dimensoes) -- um gesto cada.
    """
    centro_a, centro_b = a.mean(axis=0), b.mean(axis=0)

    # Distancia media de cada amostra ao centro do seu proprio gesto.
    # E o "raio da nuvem": o quanto o mesmo gesto varia consigo mesmo.
    intra = float(
        np.mean(
            [
                np.mean(np.linalg.norm(a - centro_a, axis=1)),
                np.mean(np.linalg.norm(b - centro_b, axis=1)),
            ]
        )
    )

    # Distancia entre os centros das duas nuvens.
    inter = float(np.linalg.norm(centro_a - centro_b))

    return intra, inter, (inter / intra if intra > 0 else 0.0)


def main() -> int:
    print("\n" + "=" * 78)
    print("ANALISE DE FEATURES -- dois gestos, movendo a mao pelo quadro")
    print("=" * 78)
    print(
        "\nVoce vai fazer DOIS gestos, 12 segundos cada. Em cada um:\n"
        "  MANTENHA o gesto, mas MOVA a mao -- esquerda, direita, perto, longe.\n"
        "\nIsso e o ponto do teste: estamos vendo qual representacao consegue\n"
        "dizer 'e o mesmo gesto' mesmo com a mao passeando pelo quadro.\n"
    )
    input("Pressione ENTER para comecar com a MAO ABERTA (palma para a camera)...")

    abertas = coletar("MAO ABERTA", "mantenha o gesto e MOVA a mao pelo quadro")
    print(f"  coletadas {len(abertas)} amostras\n")

    input("Pressione ENTER para o segundo gesto: PUNHO FECHADO...")
    punhos = coletar("PUNHO FECHADO", "mantenha o gesto e MOVA a mao pelo quadro")
    print(f"  coletadas {len(punhos)} amostras\n")

    if len(abertas) < MIN_AMOSTRAS or len(punhos) < MIN_AMOSTRAS:
        print(
            f"Amostras insuficientes ({len(abertas)} e {len(punhos)}, "
            f"minimo {MIN_AMOSTRAS} cada).\n"
            "A mao precisa ficar VISIVEL. Rode de novo.\n"
        )
        return 1

    representacoes = {
        "A) landmarks normalizados pela IMAGEM": (
            np.array([lm.ravel() for lm, _ in abertas]),
            np.array([lm.ravel() for lm, _ in punhos]),
        ),
        "B) world landmarks CRUS (metros)": (
            np.array([w.ravel() for _, w in abertas]),
            np.array([w.ravel() for _, w in punhos]),
        ),
        "C) NOSSAS features (pulso + escala)": (
            np.array([normalizar_mao(w) for _, w in abertas]),
            np.array([normalizar_mao(w) for _, w in punhos]),
        ),
    }

    print("=" * 78)
    print(f"{'representacao':<38} {'intra':>8} {'inter':>8} {'RAZAO':>8}")
    print("-" * 78)

    razoes: dict[str, float] = {}
    for nome, (a, b) in representacoes.items():
        intra, inter, razao = estatisticas(a, b)
        razoes[nome] = razao
        print(f"{nome:<38} {intra:>8.3f} {inter:>8.3f} {razao:>8.2f}")

    print("-" * 78)
    print(
        "\nintra = o quanto o MESMO gesto varia consigo mesmo (menor e melhor)\n"
        "inter = o quanto os DOIS gestos se afastam    (maior e melhor)\n"
        "RAZAO = inter / intra -> o quao FACIL e separar os dois. MAIOR E MELHOR.\n"
    )

    melhor = max(razoes, key=lambda k: razoes[k])
    print(f"Vencedor: {melhor}  (razao {razoes[melhor]:.2f})")

    a_ = razoes["A) landmarks normalizados pela IMAGEM"]
    c_ = razoes["C) NOSSAS features (pulso + escala)"]
    if a_ > 0:
        print(f"\nAs nossas features sao {c_ / a_:.1f}x mais separaveis que a opcao (A),")
        print("que e a que a maioria dos tutoriais usa.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
