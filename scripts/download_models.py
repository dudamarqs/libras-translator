"""Baixa os modelos pre-treinados do MediaPipe (arquivos .task).

    .venv\\Scripts\\python.exe scripts\\download_models.py

Por que isto e um script, e nao os arquivos commitados no Git:

Sao binarios de dezenas de MB. O Git guarda a historia COMPLETA de cada arquivo
-- e binarios nao "diferenciam": cada versao e guardada inteira. Um modelo de
30 MB atualizado 5 vezes vira 150 MB no repositorio, para sempre, mesmo depois
de voce apagar o arquivo. Modelo e ARTEFATO, nao codigo-fonte. Artefatos se
baixam; codigo se versiona.

(Na Etapa 6 vamos discutir DVC, que resolve exatamente isso para datasets.)

Por que o MediaPipe passou a exigir esse download: na API antiga (`mp.solutions`,
removida) o modelo vinha embutido no pacote. A API `tasks` separou os dois --
o que e MELHOR: o mesmo arquivo .task roda em Python, Android, iOS e no
NAVEGADOR (via @mediapipe/tasks-vision). Isso mantem aberta a otimizacao da
Etapa 16. Ver ADR-006.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "models" / "mediapipe"

BASE = "https://storage.googleapis.com/mediapipe-models"

# `float16` e uma versao QUANTIZADA do modelo: os pesos foram convertidos de
# float32 (4 bytes) para float16 (2 bytes). O arquivo cai pela metade e a
# inferencia fica mais rapida, com perda de precisao praticamente imperceptivel
# para landmarks. E o padrao recomendado pelo Google para tempo real.
MODELOS: dict[str, str] = {
    "hand_landmarker.task": (
        f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


def baixar(nome: str, url: str) -> bool:
    caminho = DESTINO / nome

    if caminho.exists():
        tamanho_mb = caminho.stat().st_size / 1024 / 1024
        sha = hashlib.sha256(caminho.read_bytes()).hexdigest()[:12]
        print(f"  [ja existe] {nome}  ({tamanho_mb:.1f} MB, sha256:{sha})")
        return True

    print(f"  [baixando]  {nome} ...", end=" ", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resposta:  # noqa: S310
            dados = resposta.read()
    except Exception as exc:  # noqa: BLE001
        print(f"FALHOU: {exc}")
        return False

    caminho.write_bytes(dados)
    sha = hashlib.sha256(dados).hexdigest()[:12]
    print(f"OK ({len(dados) / 1024 / 1024:.1f} MB, sha256:{sha})")
    return True


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    print(f"\nDestino: {DESTINO}\n")

    todos_ok = all(baixar(nome, url) for nome, url in MODELOS.items())

    if todos_ok:
        print("\nModelos prontos.\n")
        return 0
    print("\nAlgum download falhou. Verifique a conexao e rode de novo.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
