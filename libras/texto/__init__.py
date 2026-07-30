"""Camada de texto: montagem de mensagem + correcao da legenda (Etapa 10).

Expoe uma FABRICA (`criar_corretor`) que escolhe automaticamente o melhor
corretor disponivel no ambiente, do mais capaz ao mais simples:

    Claude (contexto+gramatica, pago)  >  offline (palavra, gratis)  >  cru

Assim o mesmo codigo roda de graca por padrao e usa o LLM quando ha chave, sem
`if` nenhum espalhado pelo sistema -- todos falam com a interface `Corretor`.
"""

from __future__ import annotations

import os

from libras.texto.corretor import Corretor, CorretorLLM, ErroCorretor, sdk_disponivel
from libras.texto.corretor_local import (
    CorretorLocal,
    CorretorPassthrough,
    pyspellchecker_disponivel,
)
from libras.texto.legenda import LegendaAoVivo
from libras.texto.montador import MontadorDeMensagem

__all__ = [
    "Corretor",
    "CorretorLLM",
    "CorretorLocal",
    "CorretorPassthrough",
    "ErroCorretor",
    "LegendaAoVivo",
    "MontadorDeMensagem",
    "criar_corretor",
]


def _tem_credencial_anthropic() -> bool:
    # Heuristica barata: a chave no ambiente. Nao criamos o cliente aqui (isso
    # poderia falhar por rede); so checamos se ha COMO autenticar. Um perfil
    # `ant auth login` tambem funcionaria, mas nao da pra detectar sem tentar.
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def criar_corretor(*, preferir_llm: bool = True) -> Corretor:
    """Devolve o melhor corretor que o ambiente permite.

    preferir_llm=False forca a opcao gratuita mesmo havendo chave (util para
    NAO gastar creditos durante o desenvolvimento).
    """
    if preferir_llm and sdk_disponivel() and _tem_credencial_anthropic():
        return CorretorLLM()
    if pyspellchecker_disponivel():
        return CorretorLocal()
    return CorretorPassthrough()
