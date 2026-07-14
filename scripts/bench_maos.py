"""Benchmark do MediaPipe: quanto custa detectar maos, e o que muda o custo.

    .venv\\Scripts\\python.exe scripts\\bench_maos.py

Responde tres perguntas que voce PRECISA saber antes da Etapa 5:

1. O MediaPipe cabe nos ~33 ms de folga que medimos na Etapa 3? (ADR-008)
2. Quanto o modo VIDEO (com tracker) economiza em relacao ao IMAGE?
3. Quanto custa detectar 2 maos em vez de 1?

IMPORTANTE: mostre a mao para a camera enquanto isto roda. Detectar o VAZIO e
enganosamente barato -- o palm detector nao acha nada e desiste rapido.
"""

from __future__ import annotations

import time

import numpy as np

from libras.vision.camera import Camera, CameraConfig
from libras.vision.hands import DetectorMaos

N_FRAMES = 90  # ~3 segundos
N_AQUECIMENTO = 25

# Abaixo disto, os numeros nao significam nada -- ver `main()`.
MIN_PCT_COM_MAO = 80.0


def medir(*, modo_video: bool, max_maos: int) -> dict[str, float]:
    tempos: list[float] = []
    frames_com_mao = 0

    with (
        Camera(CameraConfig()) as cam,
        DetectorMaos(max_maos=max_maos, modo_video=modo_video) as detector,
    ):
        # AQUECIMENTO. A primeira inferencia carrega o modelo, aloca os buffers
        # nativos e inicializa o delegate XNNPACK -- custa varias VEZES o normal.
        # Uma primeira versao deste script aquecia so 10 frames e o primeiro
        # cenario media 55ms contra 16ms dos demais: era cold start disfarçado
        # de resultado. Se um cenario destoa MUITO dos outros, desconfie do
        # aquecimento antes de acreditar no numero.
        for _ in range(N_AQUECIMENTO):
            detector.detectar(cam.ler())

        for _ in range(N_FRAMES):
            frame = cam.ler()
            inicio = time.perf_counter()
            resultado = detector.detectar(frame)
            tempos.append((time.perf_counter() - inicio) * 1000)
            frames_com_mao += int(not resultado.vazio)

    arr = np.array(tempos)
    return {
        "media": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        # O p95 e o numero que importa em tempo real. A MEDIA esconde os picos;
        # e o pico que faz o frame estourar o orcamento e o video "engasgar".
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "pct_com_mao": 100.0 * frames_com_mao / N_FRAMES,
    }


def main() -> int:
    print("\n" + "=" * 84)
    print(">>> MOSTRE A MAO PARA A CAMERA e mantenha ate o fim (~30s). <<<")
    print("=" * 84)
    print(
        "\nPor que isto e obrigatorio, e nao uma sugestao:\n"
        "  Sem mao no quadro, o tracker do modo VIDEO nao tem NADA para rastrear --\n"
        "  entao ele roda o palm detector em todo frame, exatamente como o modo IMAGE.\n"
        "  Os dois fazem o MESMO trabalho, o benchmark mede a MESMA coisa duas vezes,\n"
        "  e a comparacao que ele existe para fazer vira zero.\n"
        "\n  Este script se recusa a concluir se nao vir mao. Um benchmark que produz um\n"
        "  numero mesmo quando a premissa nao vale e pior do que benchmark nenhum:\n"
        "  alguem vai citar esse numero depois.\n"
    )
    for i in (3, 2, 1):
        print(f"  comecando em {i}...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 30, end="\r")

    cenarios = [
        ("VIDEO (tracker ligado), 1 mao", {"modo_video": True, "max_maos": 1}),
        ("VIDEO (tracker ligado), 2 maos", {"modo_video": True, "max_maos": 2}),
        ("IMAGE (detector em TODO frame), 1 mao", {"modo_video": False, "max_maos": 1}),
        ("IMAGE (detector em TODO frame), 2 maos", {"modo_video": False, "max_maos": 2}),
    ]

    print(f"{'cenario':<40} {'media':>7} {'p50':>7} {'p95':>7} {'max':>7}  {'com mao':>8}")
    print("-" * 84)

    resultados: dict[str, dict[str, float]] = {}
    for nome, kwargs in cenarios:
        r = medir(**kwargs)  # type: ignore[arg-type]
        resultados[nome] = r
        print(
            f"{nome:<40} {r['media']:>6.1f}ms {r['p50']:>6.1f}ms "
            f"{r['p95']:>6.1f}ms {r['max']:>6.1f}ms  {r['pct_com_mao']:>7.0f}%"
        )

    print("-" * 84)

    # Guarda de validade. SEM ISTO, o script cospe um "1.1x" que parece uma
    # conclusao e e so ruido -- porque os dois modos estavam fazendo o mesmo
    # trabalho (rodar o detector num quadro vazio).
    pior_cobertura = min(r["pct_com_mao"] for r in resultados.values())
    if pior_cobertura < MIN_PCT_COM_MAO:
        print(
            f"\n[RESULTADO INVALIDO] A mao apareceu em apenas {pior_cobertura:.0f}% dos frames\n"
            f"                     de algum cenario (minimo exigido: {MIN_PCT_COM_MAO:.0f}%).\n\n"
            "Sem mao no quadro, VIDEO e IMAGE rodam o mesmo detector em todo frame e a\n"
            "comparacao nao mede nada. Rode de novo mantendo a mao visivel o tempo todo.\n"
        )
        return 1

    video = resultados["VIDEO (tracker ligado), 2 maos"]["media"]
    imagem = resultados["IMAGE (detector em TODO frame), 2 maos"]["media"]
    print(f"\nO tracker do modo VIDEO economiza {imagem / video:.1f}x sobre o modo IMAGE.")
    print("Essa e a diferenca entre rodar o palm detector (caro) em TODO frame")
    print("ou so quando o rastreamento se perde.")

    folga = 33.3
    p95 = resultados["VIDEO (tracker ligado), 2 maos"]["p95"]
    print(f"\nOrcamento por frame a 30 FPS: {folga:.1f} ms")
    print(f"MediaPipe (VIDEO, 2 maos, p95): {p95:.1f} ms")
    if p95 < folga:
        print(f"-> CABE. Sobram {folga - p95:.1f} ms para o classificador. FPS nao deve cair.\n")
    else:
        print("-> NAO CABE no p95. O FPS vai engasgar nos picos (ver ADR-008).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
