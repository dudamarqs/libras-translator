"""Testes das funcoes de data augmentation.

Funcoes puras sobre a nuvem 3D -- testaveis sem camera nem modelo. O que
importa validar aqui sao PROPRIEDADES GEOMETRICAS: uma rotacao nao pode mudar o
tamanho da mao, o espelho tem que inverter so o x, e a augmentation tem que
PRESERVAR os rotulos (uma letra rotacionada continua a mesma letra) -- se ela
trocasse um rotulo, envenenaria o treino silenciosamente.
"""

from __future__ import annotations

import numpy as np

from libras.data.augment import (
    aumentar_treino,
    espelhar_x,
    matriz_rotacao,
    rotacionar,
)
from libras.vision.hands import N_LANDMARKS


def _mao(n: int = 3) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.normal(0, 0.05, (n, N_LANDMARKS, 3)).astype(np.float32)


class TestRotacao:
    def test_rotacao_zero_nao_muda_nada(self) -> None:
        w = _mao()
        r = matriz_rotacao(0, 0, 0)
        np.testing.assert_allclose(rotacionar(w, r), w, atol=1e-6)

    def test_rotacao_preserva_distancias(self) -> None:
        # Rotacao e rigida: NAO pode esticar nem encolher a mao. Se mudasse as
        # distancias, mudaria a escala -> a normalizacao daria outro vetor ->
        # nao seria mais a mesma letra.
        w = _mao(1)[0]  # (21, 3)
        r = matriz_rotacao(15, -10, 20)
        wr = rotacionar(w, r)
        d0 = np.linalg.norm(w[8] - w[0])
        d1 = np.linalg.norm(wr[8] - wr[0])
        np.testing.assert_allclose(d1, d0, atol=1e-5)

    def test_matriz_e_ortonormal(self) -> None:
        # Propriedade que garante que e uma rotacao pura (sem escala/cisalhamento).
        r = matriz_rotacao(30, 45, -60)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-5)
        assert np.isclose(np.linalg.det(r), 1.0, atol=1e-5)


class TestEspelho:
    def test_inverte_so_o_x(self) -> None:
        w = _mao(1)[0]
        e = espelhar_x(w)
        np.testing.assert_allclose(e[:, 0], -w[:, 0], atol=1e-6)  # x invertido
        np.testing.assert_allclose(e[:, 1], w[:, 1], atol=1e-6)  # y intacto
        np.testing.assert_allclose(e[:, 2], w[:, 2], atol=1e-6)  # z intacto

    def test_espelhar_duas_vezes_volta_ao_original(self) -> None:
        w = _mao()
        np.testing.assert_allclose(espelhar_x(espelhar_x(w)), w, atol=1e-6)


class TestAumentarTreino:
    def test_expande_e_preserva_rotulos(self) -> None:
        world = _mao(10)
        rotulo = np.array(list("ABCDEFGHIJ"))
        w_aug, y_aug = aumentar_treino(world, rotulo, n_rotacoes=2, espelhar=True, semente=0)
        # original + espelho + 2 rot do original + 2 rot do espelho = 6x
        assert len(w_aug) == 6 * len(world)
        assert len(y_aug) == len(w_aug)
        # cada rotulo aparece exatamente 6 vezes -- nenhuma amostra trocou de classe
        for letra in rotulo:
            assert int(np.sum(y_aug == letra)) == 6

    def test_o_bloco_original_vem_intacto_no_inicio(self) -> None:
        # A primeira copia e o dado real, sem transformacao -- o modelo tem que
        # ver o original, nao so versoes deformadas.
        world = _mao(5)
        rotulo = np.array(list("ABCDE"))
        w_aug, y_aug = aumentar_treino(world, rotulo, n_rotacoes=1, espelhar=False)
        np.testing.assert_array_equal(w_aug[: len(world)], world)
        np.testing.assert_array_equal(y_aug[: len(world)], rotulo)

    def test_determinismo_pela_semente(self) -> None:
        world, rotulo = _mao(4), np.array(list("ABCD"))
        a, _ = aumentar_treino(world, rotulo, semente=7)
        b, _ = aumentar_treino(world, rotulo, semente=7)
        np.testing.assert_array_equal(a, b)
