"""Testes do corretor offline e da fabrica -- sem dicionario real, sem API."""

from __future__ import annotations

from libras.texto import CorretorLocal, CorretorPassthrough, criar_corretor
from libras.texto.corretor_local import CorretorLocal as CorretorLocalDireto


# Um "dicionario" falso: corrige palavra por palavra de forma deterministica.
def _corretor_falso(palavra: str) -> str:
    mapa = {"cauro": "carro", "oi": "oi", "tudo": "tudo"}
    baixa = palavra.lower()
    return mapa.get(baixa, palavra.upper())  # desconhecida -> maiuscula


class TestCorretorLocal:
    def test_corrige_palavra_por_palavra(self) -> None:
        c = CorretorLocal(corrigir_palavra=_corretor_falso)
        assert c.corrigir("CAURO") == "carro"

    def test_preserva_a_estrutura_de_palavras(self) -> None:
        c = CorretorLocal(corrigir_palavra=_corretor_falso)
        assert c.corrigir("OI TUDO") == "oi tudo"

    def test_palavra_desconhecida_vira_maiuscula(self) -> None:
        # Nome proprio que nao esta em dicionario nenhum: nao inventamos, so
        # mantemos a soletracao. Datilologia serve justamente para nomes.
        c = CorretorLocal(corrigir_palavra=_corretor_falso)
        assert c.corrigir("XYZZY") == "XYZZY"

    def test_ignora_o_contexto_sem_quebrar(self) -> None:
        # A assinatura aceita `contexto` (para cumprir a interface Corretor),
        # mas a versao offline nao o usa. Passar contexto nao pode dar erro.
        c = CorretorLocal(corrigir_palavra=_corretor_falso)
        assert c.corrigir("CAURO", contexto="Eu comprei um") == "carro"

    def test_texto_vazio(self) -> None:
        c = CorretorLocal(corrigir_palavra=_corretor_falso)
        assert c.corrigir("   ") == ""


class TestPassthrough:
    def test_devolve_o_texto_cru(self) -> None:
        assert CorretorPassthrough().corrigir("C A S A") == "C A S A"


class TestFabrica:
    def test_sem_credencial_nem_dicionario_cai_no_passthrough(self, monkeypatch) -> None:
        # Simula um ambiente pelado: sem chave, sem SDK, sem pyspellchecker.
        import libras.texto as texto

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(texto, "sdk_disponivel", lambda: False)
        monkeypatch.setattr(texto, "pyspellchecker_disponivel", lambda: False)

        assert isinstance(criar_corretor(), CorretorPassthrough)

    def test_com_dicionario_mas_sem_chave_usa_o_local(self, monkeypatch) -> None:
        import libras.texto as texto

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(texto, "pyspellchecker_disponivel", lambda: True)

        assert isinstance(criar_corretor(), CorretorLocalDireto)

    def test_preferir_llm_falso_ignora_a_chave(self, monkeypatch) -> None:
        # Durante o desenvolvimento, forcar o gratuito mesmo tendo chave, para
        # nao gastar creditos.
        import libras.texto as texto

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-teste")
        monkeypatch.setattr(texto, "sdk_disponivel", lambda: True)
        monkeypatch.setattr(texto, "pyspellchecker_disponivel", lambda: True)

        corretor = criar_corretor(preferir_llm=False)
        assert isinstance(corretor, CorretorLocalDireto)
