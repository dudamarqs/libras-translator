"""Testes do corretor por IA -- com um cliente FALSO, sem tocar na API.

O ponto: verificar que montamos a requisicao certa (modelo, prompt, contexto) e
que extraimos a resposta certa (ignorando blocos de 'thinking') SEM gastar um
centavo nem precisar de chave. E o valor da injecao de dependencia.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from libras.texto.corretor import CorretorLLM, ErroCorretor, _extrair_texto


# --- dubles com a forma da resposta do SDK Anthropic ------------------------
@dataclass
class BlocoFalso:
    type: str
    text: str = ""


@dataclass
class RespostaFalsa:
    content: list[BlocoFalso]


@dataclass
class MessagesFalso:
    resposta: RespostaFalsa
    chamadas: list[dict] = field(default_factory=list)

    def create(self, **kwargs) -> RespostaFalsa:
        self.chamadas.append(kwargs)
        return self.resposta


@dataclass
class ClienteFalso:
    messages: MessagesFalso


def _cliente(texto_resposta: str, *, com_thinking: bool = False) -> ClienteFalso:
    blocos = []
    if com_thinking:
        blocos.append(BlocoFalso(type="thinking", text=""))  # o bloco que engana
    blocos.append(BlocoFalso(type="text", text=texto_resposta))
    return ClienteFalso(messages=MessagesFalso(resposta=RespostaFalsa(content=blocos)))


class TestCorrigir:
    def test_devolve_o_texto_do_bloco_de_texto(self) -> None:
        corretor = CorretorLLM(cliente=_cliente("carro"))
        assert corretor.corrigir("CAURO") == "carro"

    def test_ignora_bloco_de_thinking(self) -> None:
        # Se a rede pensar, o primeiro bloco e 'thinking' (texto vazio). Ler
        # content[0].text as cegas devolveria "" -- por isso filtramos por type.
        corretor = CorretorLLM(cliente=_cliente("Oi, tudo bem?", com_thinking=True))
        assert corretor.corrigir("OI TUDO BM") == "Oi, tudo bem?"

    def test_texto_vazio_nem_chama_o_llm(self) -> None:
        cliente = _cliente("nao deveria ser chamado")
        corretor = CorretorLLM(cliente=cliente)
        assert corretor.corrigir("   ") == ""
        assert cliente.messages.chamadas == []  # zero chamadas -> zero custo

    def test_usa_o_modelo_e_o_prompt_de_sistema(self) -> None:
        cliente = _cliente("casa")
        corretor = CorretorLLM(cliente=cliente, modelo="claude-opus-5")
        corretor.corrigir("CASA")

        (chamada,) = cliente.messages.chamadas
        assert chamada["model"] == "claude-opus-5"
        assert "datilologia" in chamada["system"].lower()
        assert chamada["messages"][0]["role"] == "user"
        assert "CASA" in chamada["messages"][0]["content"]

    def test_contexto_entra_na_mensagem(self) -> None:
        cliente = _cliente("carro")
        corretor = CorretorLLM(cliente=cliente)
        corretor.corrigir("CAURO", contexto="Eu comprei um")

        conteudo = cliente.messages.chamadas[0]["messages"][0]["content"]
        assert "Eu comprei um" in conteudo
        assert "CAURO" in conteudo

    def test_sem_contexto_nao_menciona_contexto(self) -> None:
        cliente = _cliente("casa")
        corretor = CorretorLLM(cliente=cliente)
        corretor.corrigir("CASA")
        conteudo = cliente.messages.chamadas[0]["messages"][0]["content"]
        assert "Contexto" not in conteudo


class TestFalhas:
    def test_erro_do_llm_vira_ErroCorretor(self) -> None:
        class ClienteQueFalha:
            class messages:  # noqa: N801
                @staticmethod
                def create(**_):
                    raise ConnectionError("rede fora")

        corretor = CorretorLLM(cliente=ClienteQueFalha())
        with pytest.raises(ErroCorretor, match="falha ao chamar"):
            corretor.corrigir("CASA")


class TestExtrairTexto:
    def test_resposta_sem_content_devolve_vazio(self) -> None:
        assert _extrair_texto(object()) == ""

    def test_junta_multiplos_blocos_de_texto(self) -> None:
        resp = RespostaFalsa(content=[BlocoFalso("text", "Oi. "), BlocoFalso("text", "Tudo bem?")])
        assert _extrair_texto(resp) == "Oi. Tudo bem?"
