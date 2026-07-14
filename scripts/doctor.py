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

    # Testar tensor nao basta -- o que precisa funcionar e o AUTOGRAD, o motor
    # que calcula os gradientes. Sem ele nao existe treinamento.
    #
    # requires_grad=True diz: "rastreie todas as operacoes feitas sobre x".
    # O torch monta um grafo dessas operacoes; .backward() percorre esse grafo
    # de tras pra frente aplicando a regra da cadeia (isso E o backpropagation)
    # e deposita d(perda)/dx dentro de x.grad.
    x = torch.randn(4, 3, requires_grad=True)
    perda = (x @ x.T).sum()
    perda.backward()

    if x.grad is None or x.grad.shape != x.shape:
        erro("autograd nao preencheu os gradientes -- treinamento nao vai funcionar")
    else:
        ok(f"tensores + autograd OK (backward preencheu x.grad, shape {tuple(x.grad.shape)})")

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
secao("5. MediaPipe: API de visao")
# ---------------------------------------------------------------------------
# O MediaPipe TINHA duas APIs:
#   - "solutions" (legada): mp.solutions.hands.Hands() -- REMOVIDA na 0.10.35.
#   - "tasks" (atual):      HandLandmarker + arquivo .task baixado a parte.
# Quase todo tutorial de Libras na internet usa a `solutions` e NAO roda aqui.
# Ver ADR-006. Este check existe para que isso nunca mais seja surpresa.
try:
    import mediapipe as mp

    if hasattr(mp, "solutions"):
        aviso(
            "esta versao ainda expoe a API legada `mp.solutions`. "
            "Ignore-a: o projeto usa `mp.tasks` (ver ADR-006)."
        )

    from mediapipe.tasks.python import vision

    for classe in ("HandLandmarker", "PoseLandmarker", "HolisticLandmarker"):
        if hasattr(vision, classe):
            ok(f"mp.tasks.python.vision.{classe} disponivel")
        else:
            erro(f"mp.tasks.python.vision.{classe} AUSENTE")
except Exception as exc:  # noqa: BLE001
    erro(f"mediapipe: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
secao("6. Um unico pacote de OpenCV")
# ---------------------------------------------------------------------------
# `opencv-python` e `opencv-contrib-python` instalam o MESMO modulo cv2/.
# Com os dois instalados, um sobrescreve o outro no disco e o ambiente fica
# num estado misto: o pip diz uma versao, o `import cv2` usa outra. Ver ADR-005.
try:
    from importlib.metadata import distributions

    instalados = sorted(
        d.metadata["Name"]
        for d in distributions()
        if (d.metadata["Name"] or "").startswith("opencv")
    )
    if len(instalados) == 1:
        ok(f"apenas um pacote OpenCV instalado: {instalados[0]}")
    elif len(instalados) == 0:
        erro("nenhum pacote OpenCV encontrado")
    else:
        erro(
            f"MAIS DE UM pacote OpenCV instalado: {instalados}. "
            "Eles se sobrescrevem. Desinstale TODOS e reinstale so o "
            "opencv-contrib-python (ver ADR-005)."
        )
except Exception as exc:  # noqa: BLE001
    erro(f"checagem de opencv: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
secao("7. Webcam")
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
secao("8. Pacote `libras` importavel")
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
