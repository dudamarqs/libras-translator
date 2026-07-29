"""Testes do montador de mensagem -- logica pura, sem camera nem IA."""

from __future__ import annotations

from libras.texto.montador import MontadorDeMensagem


class TestMontagem:
    def test_letras_viram_palavra(self) -> None:
        m = MontadorDeMensagem()
        for c in "CASA":
            m.letra(c)
        assert m.texto_bruto() == "CASA"
        assert m.palavra_em_construcao == "CASA"

    def test_pausa_separa_palavras(self) -> None:
        m = MontadorDeMensagem()
        for c in "OI":
            m.letra(c)
        m.fim_de_palavra()
        for c in "TUDO":
            m.letra(c)
        assert m.texto_bruto() == "OI TUDO"

    def test_pausas_repetidas_nao_criam_espacos_duplos(self) -> None:
        # A mao fica varios frames fora do quadro -> varias chamadas a
        # fim_de_palavra(). Nao pode virar "OI   TUDO".
        m = MontadorDeMensagem()
        for c in "OI":
            m.letra(c)
        m.fim_de_palavra()
        m.fim_de_palavra()
        m.fim_de_palavra()
        for c in "TUDO":
            m.letra(c)
        assert m.texto_bruto() == "OI TUDO"

    def test_pausa_no_inicio_nao_faz_nada(self) -> None:
        m = MontadorDeMensagem()
        m.fim_de_palavra()
        assert m.vazio
        assert m.texto_bruto() == ""


class TestApagar:
    def test_apaga_ultima_letra(self) -> None:
        m = MontadorDeMensagem()
        for c in "CAX":  # soletrou C, A, e errou com X
            m.letra(c)
        m.apagar()  # tira o X
        m.letra("S")
        m.letra("A")
        assert m.texto_bruto() == "CASA"

    def test_apaga_reabre_palavra_anterior(self) -> None:
        # Palavra atual vazia -> apagar volta a editar a ultima palavra fechada.
        m = MontadorDeMensagem()
        for c in "OI":
            m.letra(c)
        m.fim_de_palavra()
        m.apagar()  # nao ha palavra atual -> reabre "OI"
        assert m.palavra_em_construcao == "OI"
        m.apagar()  # tira o I
        assert m.texto_bruto() == "O"

    def test_apagar_vazio_nao_quebra(self) -> None:
        m = MontadorDeMensagem()
        m.apagar()  # nao deve levantar excecao
        assert m.vazio


class TestLimpar:
    def test_limpar_zera_tudo(self) -> None:
        m = MontadorDeMensagem()
        for c in "OI":
            m.letra(c)
        m.fim_de_palavra()
        m.letra("T")
        m.limpar()
        assert m.vazio
        assert m.texto_bruto() == ""
