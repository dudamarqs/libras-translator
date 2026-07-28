"""Data augmentation: sintetizar variacao que o dataset nao tem.

POR QUE, e escolhido por DIAGNOSTICO (nao por chute):

Medimos que os pares confusos (T/F, R/U) sao 99-100% separaveis DENTRO de uma
sessao e caem para 70-80% ENTRE sessoes. Logo as features nao sao cegas -- o
problema e a variacao ENTRE sessoes (a mao inclinada um pouco diferente a cada
dia). Augmentation cria essa variacao artificialmente, para o modelo aprender a
toleral-a a partir das sessoes que ja temos.

DUAS transformacoes, ambas GEOMETRICAMENTE VALIDAS sobre a nuvem 3D metrica
(`world`), ao contrario de mexer em pixels:

1. ROTACAO pequena (+-graus). Simula a mao segurada com leve inclinacao
   diferente. NAO viola o ADR-012 (nao normalizamos rotacao porque orientacao da
   palma e gramatica): rotacoes PEQUENAS ensinam TOLERANCIA a tremor de angulo,
   sem apagar distincoes de orientacao, que sao de dezenas de graus.

2. ESPELHAMENTO no eixo x. Uma mao direita espelhada VIRA uma mao esquerda
   fazendo a mesma letra. Cobre canhotos e destros de graca -- resolve a
   limitacao conhecida (modelo so via a mao direita) sem coletar nada.

Aplica-se SO no conjunto de TREINO, nunca no teste (aumentar o teste seria
avaliar o modelo em dados que voce inventou).
"""

from __future__ import annotations

import numpy as np

# Espelhar o eixo x: (x, y, z) -> (-x, y, z). Troca a lateralidade da mao.
_ESPELHO_X = np.array([-1.0, 1.0, 1.0], dtype=np.float32)


def matriz_rotacao(graus_x: float, graus_y: float, graus_z: float) -> np.ndarray:
    """Matriz de rotacao 3D (composicao das rotacoes em x, y, z)."""
    rx, ry, rz = np.radians([graus_x, graus_y, graus_z])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return (mz @ my @ mx).astype(np.float32)


def rotacionar(world: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Aplica a rotacao r a landmarks (N, 21, 3) ou (21, 3)."""
    return (world @ r.T).astype(np.float32)


def espelhar_x(world: np.ndarray) -> np.ndarray:
    """Espelha a mao no eixo x (direita <-> esquerda)."""
    return (world * _ESPELHO_X).astype(np.float32)


def aumentar_treino(
    world: np.ndarray,
    rotulo: np.ndarray,
    *,
    n_rotacoes: int = 2,
    max_graus: float = 12.0,
    espelhar: bool = True,
    jitter: float = 0.002,
    semente: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Expande (world, rotulo) do TREINO com copias rotacionadas/espelhadas.

    Retorna world_aumentado (M, 21, 3) e rotulo_aumentado (M,), com M > N.
    Os rotulos sao PRESERVADOS: uma letra rotacionada ou espelhada continua a
    mesma letra.

    jitter: ruido gaussiano pequeno (em metros) somado aos pontos, simulando a
    imprecisao de deteccao do MediaPipe. Da robustez de graca.
    """
    rng = np.random.default_rng(semente)

    def _com_jitter(w: np.ndarray) -> np.ndarray:
        if jitter <= 0:
            return w
        return (w + rng.normal(0, jitter, w.shape)).astype(np.float32)

    versoes_world = [world]
    versoes_rotulo = [rotulo]

    # bases = original e (opcionalmente) espelhada; rotacionamos ambas.
    bases = [world]
    if espelhar:
        bases.append(espelhar_x(world))
        versoes_world.append(espelhar_x(world))
        versoes_rotulo.append(rotulo)

    for base in bases:
        for _ in range(n_rotacoes):
            angs = rng.uniform(-max_graus, max_graus, size=3)
            r = matriz_rotacao(*angs)
            versoes_world.append(_com_jitter(rotacionar(base, r)))
            versoes_rotulo.append(rotulo)

    return (
        np.concatenate(versoes_world).astype(np.float32),
        np.concatenate(versoes_rotulo),
    )
