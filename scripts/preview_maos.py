"""Preview da webcam com deteccao de maos do MediaPipe.

    .venv\\Scripts\\python.exe scripts\\preview_maos.py

Teclas:  q / ESC = sair   |   i = mostrar os indices dos 21 landmarks
         w = imprimir no terminal os numeros que o modelo vai receber

O que observar:

1. O `mediapipe: X ms` no HUD. Compare com os ~33 ms de folga que medimos na
   Etapa 3 (ADR-008). Enquanto couber, o FPS NAO cai.

2. Tire a mao do quadro e coloque de volta. Repare que ela e reencontrada quase
   instantaneamente -- e que o custo do MediaPipe da um PICO nesse frame. Esse
   pico e o palm detector (o modelo caro) rodando. Nos frames em que a mao ja
   esta rastreada, so o modelo barato roda.

3. Aperte `i` e olhe os indices: 0 = pulso, e cada dedo vai da base ate a ponta
   (4, 8, 12, 16, 20 sao as pontas). Sao esses 21 pontos que viram as features.
"""

from __future__ import annotations

import cv2

from libras.vision.camera import Camera, CameraConfig, CameraError
from libras.vision.fps import Cronometro, MedidorFPS
from libras.vision.hands import NOMES_LANDMARKS, DetectorMaos
from libras.vision.overlay import AMARELO, VERDE, desenhar_maos, painel


def dump_landmarks(resultado) -> None:  # noqa: ANN001
    """Imprime os numeros que o classificador vai receber. Tecla `w`."""
    if resultado.vazio:
        print("\n(nenhuma mao no quadro)")
        return

    for mao in resultado.maos:
        print(
            f"\n--- mao {mao.lado} "
            f"(o MediaPipe viu '{mao.lado_bruto}' no frame espelhado; "
            f"confianca {mao.confianca_lado:.2f}) ---"
        )
        print(f"{'idx':>3} {'nome':<16} {'normalizado (x, y, z)':<28} world em metros (x, y, z)")
        for i in (0, 4, 8, 12, 16, 20):  # pulso + as 5 pontas
            n = mao.landmarks[i]
            w = mao.world[i]
            print(
                f"{i:>3} {NOMES_LANDMARKS[i]:<16} "
                f"({n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f})      "
                f"({w[0]:+.3f}, {w[1]:+.3f}, {w[2]:+.3f})"
            )
        print(f"    shape: landmarks {mao.landmarks.shape}  world {mao.world.shape}")
        print(f"    -> uma mao = {mao.world.size} numeros (21 pontos x 3 eixos)")


def main() -> int:
    fps = MedidorFPS(janela=30)
    crono = Cronometro(janela=30)
    mostrar_indices = False

    config = CameraConfig()

    try:
        with (
            Camera(config) as cam,
            # `entrada_espelhada` DERIVADO da config da camera, nao repetido a
            # mao. Se alguem desligar o espelhamento e esquecer de mexer aqui,
            # os rotulos "Left"/"Right" sairiam trocados -- sem erro nenhum,
            # so contaminando o dataset. Amarrar os dois torna isso impossivel.
            DetectorMaos(max_maos=2, entrada_espelhada=config.espelhar) as detector,
        ):
            print("q/ESC = sair | i = indices | w = imprimir landmarks\n")

            while True:
                with crono.medir("captura"):
                    frame = cam.ler()

                with crono.medir("mediapipe"):
                    resultado = detector.detectar(frame)

                with crono.medir("desenho"):
                    desenhar_maos(frame, resultado.maos, mostrar_indices=mostrar_indices)

                fps.marcar()
                t = crono.resumo()

                painel(
                    frame,
                    [
                        f"FPS: {fps.fps:5.1f}",
                        f"mediapipe: {t.get('mediapipe', 0):5.1f} ms   "
                        f"desenho: {t.get('desenho', 0):4.1f} ms",
                        f"maos detectadas: {len(resultado.maos)}",
                    ],
                    cor=VERDE if not resultado.vazio else AMARELO,
                )

                cv2.imshow("Libras - Etapa 4: maos", frame)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), 27):
                    break
                if tecla == ord("i"):
                    mostrar_indices = not mostrar_indices
                if tecla == ord("w"):
                    dump_landmarks(resultado)

    except CameraError as exc:
        print(f"\nERRO DE CAMERA: {exc}")
        return 1
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return 1
    except KeyboardInterrupt:
        print("\ninterrompido")

    print("\ntempos medios (ms):", {k: round(v, 1) for k, v in crono.resumo().items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
