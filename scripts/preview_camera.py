"""Preview da webcam com HUD de diagnostico.

    .venv\\Scripts\\python.exe scripts\\preview_camera.py

Teclas:  q ou ESC = sair   |   e = liga/desliga o espelhamento

Objetivo desta etapa: provar que o loop de captura esta solido e MEDIDO antes de
qualquer IA entrar nele.

Como LER o numero `captura: X ms`: ele NAO e CPU gasta -- e o tempo BLOQUEADO
esperando a camera. Com a camera a 30 FPS, ele fica em ~33ms, que e simplesmente
o periodo entre frames. O loop se comporta assim:

    tempo_do_frame  ~=  max( periodo_da_camera , tempo_de_processamento )

Ou seja: enquanto o processamento couber em 33ms, ele e DE GRACA -- o FPS nao
cai nada. Essa e a folga que o MediaPipe vai ocupar na Etapa 4. Ver ADR-008.
"""

from __future__ import annotations

import cv2

from libras.vision.camera import Camera, CameraConfig, CameraError
from libras.vision.fps import Cronometro, MedidorFPS
from libras.vision.overlay import VERDE, painel


def main() -> int:
    config = CameraConfig(indice=0, largura=640, altura=480, espelhar=True)
    fps = MedidorFPS(janela=30)
    crono = Cronometro(janela=30)

    try:
        with Camera(config) as cam:
            largura, altura = cam.resolucao_efetiva
            print(f"camera aberta em {largura}x{altura} (pedimos {config.largura}x{config.altura})")
            print("q/ESC = sair | e = espelhar")

            espelhar = config.espelhar

            while True:
                with crono.medir("captura"):
                    frame = cam.ler()

                # O `ler()` ja espelha conforme a config. Este toggle serve para
                # voce VER a diferenca -- e entender por que a convencao importa.
                if espelhar != config.espelhar:
                    frame = cv2.flip(frame, 1)

                fps.marcar()

                tempos = crono.resumo()
                painel(
                    frame,
                    [
                        f"FPS: {fps.fps:5.1f}",
                        f"captura: {tempos.get('captura', 0):5.1f} ms (bloqueado esperando)",
                        f"frame: {frame.shape[1]}x{frame.shape[0]}  espelhado: {espelhar}",
                    ],
                    cor=VERDE,
                )

                cv2.imshow("Libras - Etapa 3: captura", frame)

                # waitKey(1) NAO e "espere 1ms". E o unico ponto em que o OpenCV
                # processa os eventos da janela (redesenhar, mover, fechar).
                # Sem ele, a janela abre CINZA e congelada -- e todo iniciante
                # acha que travou. O `& 0xFF` descarta bits altos que alguns
                # sistemas colocam no codigo da tecla.
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), 27):  # 27 = ESC
                    break
                if tecla == ord("e"):
                    espelhar = not espelhar

    except CameraError as exc:
        print(f"\nERRO DE CAMERA: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\ninterrompido")

    # Repare que nao ha `cam.release()` aqui. O context manager ja fez isso --
    # inclusive se o loop tivesse morrido com uma excecao la dentro.
    print("\nresumo final de tempos (ms):", crono.resumo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
