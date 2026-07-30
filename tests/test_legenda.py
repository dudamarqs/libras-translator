"""Testes da legenda ao vivo -- concorrencia, sem camera e sem IA de verdade.

Todo corretor aqui e FALSO. O que esta sob teste nao e a qualidade da correcao
(isso e do test_corretor*), e sim o CONTRATO DE TEMPO: o loop nunca bloqueia, o
resultado certo chega, e o resultado errado (obsoleto) e descartado.

Testar codigo com thread da errado quando se usa `sleep` para "esperar dar
tempo" -- fica lento e falha aleatoriamente numa maquina carregada. Aqui usamos
duas ferramentas deterministicas: `Event` para segurar o corretor exatamente
onde queremos, e `aguardar()` para esperar o fim do trabalho de verdade.
"""

from __future__ import annotations

import threading

import pytest

from libras.texto.corretor import ErroCorretor
from libras.texto.legenda import LegendaAoVivo


class CorretorFalso:
    """Corretor instantaneo: poe a saida em minusculas e conta as chamadas."""

    def __init__(self) -> None:
        self.chamadas: list[tuple[str, str | None]] = []

    def corrigir(self, texto_bruto: str, contexto: str | None = None) -> str:
        self.chamadas.append((texto_bruto, contexto))
        return texto_bruto.lower()


class CorretorBloqueado:
    """Corretor que so responde quando o teste mandar (simula a rede lenta)."""

    def __init__(self) -> None:
        self.entrou = threading.Event()
        self.liberar = threading.Event()
        self.chamadas = 0

    def corrigir(self, texto_bruto: str, contexto: str | None = None) -> str:
        self.chamadas += 1
        self.entrou.set()
        self.liberar.wait(timeout=5.0)
        return texto_bruto.lower()


class CorretorQueFalha:
    def corrigir(self, texto_bruto: str, contexto: str | None = None) -> str:
        raise ErroCorretor("rede fora")


@pytest.fixture
def falso() -> CorretorFalso:
    return CorretorFalso()


class TestMontagem:
    def test_letras_e_pausa_viram_texto_corrigido(self, falso: CorretorFalso) -> None:
        with LegendaAoVivo(falso) as legenda:
            for c in "CASA":
                legenda.letra(c)
            assert legenda.texto_bruto == "CASA"
            legenda.fim_de_palavra()
            assert legenda.aguardar(timeout=5.0)
            assert legenda.legenda == "casa"

    def test_letra_sozinha_nao_gasta_correcao(self, falso: CorretorFalso) -> None:
        # Corrigir a cada letra seria pagar 4 chamadas ao LLM para escrever
        # "CASA". So a pausa dispara.
        with LegendaAoVivo(falso) as legenda:
            for c in "CASA":
                legenda.letra(c)
            assert legenda.aguardar(timeout=5.0)
            assert falso.chamadas == []

    def test_pausas_repetidas_nao_gastam_correcao(self, falso: CorretorFalso) -> None:
        # O loop chama fim_de_palavra() a cada frame sem mao: 30x por segundo.
        with LegendaAoVivo(falso) as legenda:
            for c in "OI":
                legenda.letra(c)
            for _ in range(30):
                legenda.fim_de_palavra()
            assert legenda.aguardar(timeout=5.0)
            assert len(falso.chamadas) == 1

    def test_apagar_desfaz_a_ultima_letra(self, falso: CorretorFalso) -> None:
        with LegendaAoVivo(falso) as legenda:
            for c in "CASAX":
                legenda.letra(c)
            legenda.apagar()
            assert legenda.texto_bruto == "CASA"


class TestNaoBloqueia:
    def test_o_loop_continua_com_o_corretor_travado(self) -> None:
        """O contrato central: correcao lenta NAO trava quem desenha o frame."""
        lento = CorretorBloqueado()
        with LegendaAoVivo(lento) as legenda:
            for c in "CASA":
                legenda.letra(c)
            legenda.fim_de_palavra()
            assert lento.entrou.wait(timeout=5.0)

            # O corretor esta preso la dentro. Tudo que o loop faz por frame
            # tem que responder mesmo assim -- se algo aqui travar, o teste
            # estoura no timeout do pytest em vez de passar.
            assert legenda.corrigindo is True
            assert legenda.texto == "CASA"  # mostra o bruto enquanto espera
            for c in "OI":
                legenda.letra(c)
            assert legenda.texto_bruto == "CASA OI"

            lento.liberar.set()
            assert legenda.aguardar(timeout=5.0)
            assert legenda.legenda == "casa"
            assert legenda.corrigindo is False


class TestResultadoObsoleto:
    def test_limpar_descarta_correcao_em_voo(self) -> None:
        """Apagar a tela nao pode ser desfeito por um resultado atrasado."""
        lento = CorretorBloqueado()
        with LegendaAoVivo(lento) as legenda:
            for c in "CASA":
                legenda.letra(c)
            legenda.fim_de_palavra()
            assert lento.entrou.wait(timeout=5.0)

            legenda.limpar()  # usuaria apertou "limpar" com o Claude pensando
            lento.liberar.set()
            assert legenda.aguardar(timeout=5.0)

            # Se a geracao nao existisse, "casa" reapareceria do nada aqui.
            assert legenda.legenda == ""
            assert legenda.texto == ""

    def test_nova_frase_descarta_correcao_em_voo(self) -> None:
        lento = CorretorBloqueado()
        with LegendaAoVivo(lento) as legenda:
            for c in "OI":
                legenda.letra(c)
            legenda.fim_de_palavra()
            assert lento.entrou.wait(timeout=5.0)

            legenda.nova_frase()
            lento.liberar.set()
            assert legenda.aguardar(timeout=5.0)
            assert legenda.legenda == ""


class TestContexto:
    def test_frase_fechada_vira_contexto_do_proximo_pedido(
        self, falso: CorretorFalso
    ) -> None:
        with LegendaAoVivo(falso) as legenda:
            for c in "OI":
                legenda.letra(c)
            legenda.fim_de_palavra()
            assert legenda.aguardar(timeout=5.0)
            legenda.nova_frase()
            assert legenda.contexto == "oi"  # arquivou a versao CORRIGIDA

            for c in "CAURO":
                legenda.letra(c)
            legenda.fim_de_palavra()
            assert legenda.aguardar(timeout=5.0)

        assert falso.chamadas[-1] == ("CAURO", "oi")

    def test_contexto_e_limitado_a_poucas_frases(self, falso: CorretorFalso) -> None:
        # Sessao longa nao pode mandar a conversa inteira a cada pedido: os
        # tokens (e o custo) cresceriam sem limite.
        with LegendaAoVivo(falso) as legenda:
            for palavra in ("UM", "DOIS", "TRES", "QUATRO", "CINCO"):
                for c in palavra:
                    legenda.letra(c)
                legenda.fim_de_palavra()
                assert legenda.aguardar(timeout=5.0)
                legenda.nova_frase()
            assert legenda.contexto == "tres quatro cinco"


class TestFalhas:
    def test_cai_no_fallback_quando_o_llm_falha(self, falso: CorretorFalso) -> None:
        with LegendaAoVivo(CorretorQueFalha(), fallback=falso) as legenda:
            for c in "CASA":
                legenda.letra(c)
            legenda.fim_de_palavra()
            assert legenda.aguardar(timeout=5.0)
            assert legenda.legenda == "casa"
            assert legenda.erro is None

    def test_sem_fallback_registra_o_erro_e_segue_vivo(self) -> None:
        falho = CorretorQueFalha()
        with LegendaAoVivo(falho) as legenda:
            for c in "CASA":
                legenda.letra(c)
            legenda.fim_de_palavra()
            assert legenda.aguardar(timeout=5.0)
            assert legenda.erro is not None
            assert legenda.texto == "CASA"  # nunca fica sem legenda

            # A thread nao morreu com a excecao: o proximo pedido e atendido.
            for c in "OI":
                legenda.letra(c)
            legenda.fim_de_palavra()
            assert legenda.aguardar(timeout=5.0)


class TestCicloDeVida:
    def test_fechar_e_idempotente(self, falso: CorretorFalso) -> None:
        legenda = LegendaAoVivo(falso)
        legenda.fechar()
        legenda.fechar()
        assert legenda.corrigindo is False
