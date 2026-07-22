"""Testes da fusao do MODO CORRECAO (`collect.py --corrigir`).

Por que esta operacao merece teste proprio: ela roda DEPOIS de voce ja ter
gravado na frente da camera. Um erro aqui nao da "erro de digitacao" -- ele
descarta amostras que custaram tempo real, ou pior, mantem as antigas achando
que regravou. E o tipo de bug que so aparece quando ja e tarde.

O que garantimos:
  - letra regravada fica SO com as amostras novas;
  - letra NAO regravada passa intacta (byte a byte);
  - o session_id continua o da sessao antiga (correcao sobrescreve no lugar).
"""

from __future__ import annotations

import numpy as np

from libras.data.dataset import Sessao
from libras.vision.hands import N_LANDMARKS
from training.collect import fundir_correcao


def sessao(session_id: str, rotulos: list[str], semente: int) -> Sessao:
    n = len(rotulos)
    rng = np.random.default_rng(semente)
    return Sessao(
        session_id=session_id,
        world=rng.normal(0, 0.05, (n, N_LANDMARKS, 3)).astype(np.float32),
        imagem=rng.random((n, N_LANDMARKS, 3)).astype(np.float32),
        lado=np.array(["Right"] * n),
        rotulo=np.array(rotulos),
    )


class TestFundirCorrecao:
    def test_letra_regravada_fica_so_com_as_novas(self) -> None:
        antiga = sessao("s1", ["A", "A", "B", "B", "C"], semente=1)
        nova = sessao("s2", ["A", "A", "A"], semente=2)

        fundida = fundir_correcao(antiga, nova, ["A"])

        contagem = fundida.contagem()
        assert contagem["A"] == 3  # so as novas, as 2 antigas sumiram
        assert contagem["B"] == 2  # intactas
        assert contagem["C"] == 1

    def test_letra_nao_regravada_passa_intacta(self) -> None:
        antiga = sessao("s1", ["A", "B", "C"], semente=3)
        nova = sessao("s2", ["A"], semente=4)

        fundida = fundir_correcao(antiga, nova, ["A"])

        # O B da sessao fundida tem que ser exatamente o B da antiga.
        b_antigo = antiga.world[antiga.rotulo == "B"]
        b_fundido = fundida.world[fundida.rotulo == "B"]
        assert np.array_equal(b_antigo, b_fundido)

    def test_mantem_o_session_id_da_antiga(self) -> None:
        antiga = sessao("sessao_20260101_000000", ["A", "B"], semente=5)
        nova = sessao("sessao_20260202_000000", ["A"], semente=6)

        fundida = fundir_correcao(antiga, nova, ["A"])

        assert fundida.session_id == "sessao_20260101_000000"

    def test_varias_letras_de_uma_vez(self) -> None:
        antiga = sessao("s1", ["A", "B", "C", "D", "D"], semente=7)
        nova = sessao("s2", ["A", "A", "D"], semente=8)

        fundida = fundir_correcao(antiga, nova, ["A", "D"])

        contagem = fundida.contagem()
        assert contagem == {"A": 2, "B": 1, "C": 1, "D": 1}

    def test_letra_pulada_nao_e_descartada(self) -> None:
        """Se voce pular a letra (tecla `s`), ela nao entra em `regravadas`.

        E a garantia que impede a pior perda silenciosa do modo correcao:
        descartar o antigo sem ter gravado o novo.
        """
        antiga = sessao("s1", ["A", "A", "B"], semente=9)
        nova = sessao("s2", ["B"], semente=10)

        # so B foi efetivamente regravada; A foi pulada
        fundida = fundir_correcao(antiga, nova, ["B"])

        assert fundida.contagem()["A"] == 2  # sobreviveu
        assert fundida.contagem()["B"] == 1
