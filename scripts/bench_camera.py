"""Benchmark dos backends de captura da webcam.

    .venv\\Scripts\\python.exe scripts\\bench_camera.py

Por que este script existe e esta versionado:

O backend de captura correto DEPENDE DA MAQUINA -- do driver, da webcam, do SO.
Nesta maquina o MSMF entrega 30 FPS e o DirectShow, 15. Na sua proxima maquina
pode ser o inverso. Nao existe resposta universal, existe MEDICAO.

Este script tambem e a evidencia do ADR-007: qualquer pessoa (inclusive voce
daqui a seis meses) pode rodar e conferir se a escolha ainda faz sentido.
"""

from __future__ import annotations

import time

import cv2

# CAP_PROP_FPS e a taxa que o DRIVER DIZ que vai entregar. Ele mente com
# frequencia -- por isso medimos o tempo de parede e ignoramos essa promessa.
BACKENDS: list[tuple[str, int]] = [
    ("MSMF (Media Foundation)", cv2.CAP_MSMF),
    ("DSHOW (DirectShow)", cv2.CAP_DSHOW),
    ("ANY (o que o OpenCV escolher)", cv2.CAP_ANY),
]

CODECS: list[str | None] = [None, "MJPG"]


def medir(
    backend: int,
    *,
    codec: str | None = None,
    largura: int = 640,
    altura: int = 480,
    n_frames: int = 60,
) -> dict[str, object] | None:
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        cap.release()
        return None

    if codec:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*codec))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # AQUECIMENTO. Os primeiros frames sempre mentem: o auto-exposure e o
    # auto-foco ainda estao convergindo, e o primeiro read() inclui o custo de
    # inicializacao do driver. Medir esses frames poluiria a media.
    for _ in range(10):
        cap.read()

    inicio = time.perf_counter()
    lidos = 0
    for _ in range(n_frames):
        sucesso, _frame = cap.read()
        lidos += int(sucesso)
    decorrido = time.perf_counter() - inicio

    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_real = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip()
    resultado = {
        "fps": lidos / decorrido if decorrido > 0 else 0.0,
        "ms": decorrido / n_frames * 1000,
        "codec": codec_real or "?",
        "resolucao": (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    }
    cap.release()
    return resultado


def main() -> int:
    print("\nMedindo... (cada combinacao le 60 frames + 10 de aquecimento)\n")
    print(f"{'combinacao':<42} {'FPS':>6} {'ms/frame':>9}  {'codec':<6} resolucao")
    print("-" * 84)

    melhor: tuple[float, str] = (0.0, "")

    for nome, backend in BACKENDS:
        for codec in CODECS:
            rotulo = f"{nome}{' + ' + codec if codec else ''}"
            r = medir(backend, codec=codec)
            if r is None:
                print(f"{rotulo:<42} {'--':>6}  (nao abriu)")
                continue
            w, h = r["resolucao"]  # type: ignore[misc]
            print(f"{rotulo:<42} {r['fps']:>6.1f} {r['ms']:>9.1f}  {r['codec']:<6} {w}x{h}")
            if r["fps"] > melhor[0]:  # type: ignore[operator]
                melhor = (float(r["fps"]), rotulo)  # type: ignore[arg-type]

    print("-" * 84)
    print(f"\nMelhor: {melhor[1]} -> {melhor[0]:.1f} FPS")
    print(
        "\nA 30 FPS o periodo da camera e 33.3 ms. Enquanto o processamento (MediaPipe +\n"
        "modelo) couber nesse periodo, ele NAO custa FPS nenhum -- o loop se comporta como\n"
        "max(periodo_da_camera, tempo_de_processamento), nao como soma. Ver ADR-008."
    )
    print("\nSe o vencedor aqui NAO for o backend padrao de CameraConfig, atualize-o.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
