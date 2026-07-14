"""Diagnostico do ambiente.

Roda:  .venv\\Scripts\\python.exe scripts\\doctor.py

Por que este script existe: em ML, "o pacote instalou" e "o pacote funciona"
sao coisas diferentes. Um numpy incompativel so estoura quando voce importa
o cv2; a webcam so falha quando outro app ja a segurou; o MediaPipe so
reclama do modelo quando voce tenta o primeiro frame.

Descobrir isso agora custa 10 segundos. Descobrir no meio da coleta de dados
custa a sessao inteira.
"""

from __future__ import annotations

import platform
import sys

VERDE = "\033[92m"
VERMELHO = "\033[91m"
AMARELO = "\033[93m"
RESET = "\033[0m"

falhas = 0


def ok(msg: str) -> None:
    print(f"  {VERDE}[OK]{RESET}   {msg}")


def erro(msg: str) -> None:
    global falhas
    falhas += 1
    print(f"  {VERMELHO}[FALHA]{RESET} {msg}")


def aviso(msg: str) -> None:
    print(f"  {AMARELO}[AVISO]{RESET} {msg}")


def secao(titulo: str) -> None:
    print(f"\n{titulo}")
    print("-" * len(titulo))


# ---------------------------------------------------------------------------
secao("1. Interpretador")
# ---------------------------------------------------------------------------
py = sys.version_info
print(f"  Python  : {platform.python_version()}")
print(f"  Executavel: {sys.executable}")

if (py.major, py.minor) == (3, 12):
    ok("Python 3.12 (versao alvo do projeto)")
else:
    erro(
        f"Esperado Python 3.12, encontrado {py.major}.{py.minor}. "
        "O MediaPipe nao publica wheels para 3.13/3.14. "
        "Voce provavelmente esta rodando fora do .venv."
    )

if ".venv" not in sys.executable:
    aviso("Este interpretador nao parece ser o do .venv do projeto.")

# ---------------------------------------------------------------------------
secao("2. Bibliotecas")
# ---------------------------------------------------------------------------
for nome, modulo in [
    ("numpy", "numpy"),
    ("opencv", "cv2"),
    ("mediapipe", "mediapipe"),
    ("torch", "torch"),
    ("scikit-learn", "sklearn"),
    ("pandas", "pandas"),
    ("fastapi", "fastapi"),
]:
    try:
        mod = __import__(modulo)
        versao = getattr(mod, "__version__", "?")
        ok(f"{nome:<13} {versao}")
    except Exception as exc:  # noqa: BLE001 - queremos capturar qualquer falha de import
        erro(f"{nome:<13} nao importou: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
secao("3. Compatibilidade binaria numpy <-> opencv <-> mediapipe")
# ---------------------------------------------------------------------------
# O teste real nao e "importou?", e "consegue trocar arrays entre si?".
# numpy expoe uma ABI em C; cv2 e mediapipe sao COMPILADOS contra ela.
# Se as versoes divergirem, o import passa e a primeira operacao real explode.
try:
    import cv2
    import numpy as np

    img = np.zeros((48, 64, 3), dtype=np.uint8)
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert cinza.shape == (48, 64)
    ok(f"numpy {np.__version__} <-> opencv {cv2.__version__} trocam arrays sem erro")
except Exception as exc:  # noqa: BLE001
    erro(f"incompatibilidade binaria: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
secao("4. PyTorch")
# ---------------------------------------------------------------------------
try:
    import torch

    x = torch.randn(4, 3)
    y = (x @ x.T).sum()
    y.backward if x.requires_grad else None
    ok(f"tensores funcionam (device padrao: cpu, resultado shape {tuple((x @ x.T).shape)})")

    if torch.cuda.is_available():
        ok(f"GPU CUDA disponivel: {torch.cuda.get_device_name(0)}")
    else:
        aviso(
            "sem GPU CUDA -- tudo bem. Nosso classificador tem poucos milhares "
            "de parametros e treina em segundos na CPU."
        )
except Exception as exc:  # noqa: BLE001
    erro(f"torch: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
secao("5. MediaPipe: qual API esta disponivel?")
# ---------------------------------------------------------------------------
# O MediaPipe tem DUAS APIs:
#   - "solutions" (legada): mp.solutions.hands.Hands() -- simples, sem baixar modelo.
#   - "tasks" (atual):      HandLandmarker -- exige o arquivo .task, e o caminho
#                            para rodar o MESMO modelo no navegador depois.
# Precisamos saber qual a versao instalada oferece antes de escrever a Etapa 4.
try:
    import mediapipe as mp

    tem_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
    tem_tasks = hasattr(mp, "tasks") and hasattr(mp.tasks, "vision")

    if tem_solutions:
        ok("API 'solutions' (legada) disponivel: mp.solutions.hands")
    else:
        aviso("API 'solutions' (legada) NAO disponivel nesta versao")

    if tem_tasks:
        ok("API 'tasks' (atual) disponivel: mp.tasks.vision.HandLandmarker")
    else:
        aviso("API 'tasks' NAO disponivel nesta versao")

    if not tem_solutions and not tem_tasks:
        erro("nenhuma API de visao do MediaPipe disponivel")
except Exception as exc:  # noqa: BLE001
    erro(f"mediapipe: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
secao("6. Webcam")
# ---------------------------------------------------------------------------
try:
    import cv2

    # CAP_DSHOW = DirectShow. No Windows, o backend padrao (MSMF) costuma levar
    # ~2s para abrir a camera e as vezes trava. DirectShow abre instantaneo.
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        erro(
            "nao consegui abrir a webcam (indice 0). "
            "Feche Teams/Zoom/Meet -- no Windows a camera e exclusiva."
        )
    else:
        sucesso, frame = cam.read()
        if not sucesso or frame is None:
            erro("a camera abriu mas nao entregou frame")
        else:
            altura, largura = frame.shape[:2]
            fps = cam.get(cv2.CAP_PROP_FPS)
            ok(f"webcam entregou frame de {largura}x{altura} (FPS reportado: {fps:.0f})")
    cam.release()
except Exception as exc:  # noqa: BLE001
    erro(f"webcam: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
secao("7. Pacote `libras` importavel")
# ---------------------------------------------------------------------------
try:
    import libras

    ok(f"import libras -> versao {libras.__version__}")
except ImportError:
    erro("`import libras` falhou. Rode:  .venv\\Scripts\\python.exe -m pip install -e .")

# ---------------------------------------------------------------------------
print()
if falhas == 0:
    print(f"{VERDE}Ambiente pronto.{RESET} Nenhuma falha.\n")
    sys.exit(0)
else:
    print(f"{VERMELHO}{falhas} falha(s).{RESET} Corrija antes de seguir.\n")
    sys.exit(1)
