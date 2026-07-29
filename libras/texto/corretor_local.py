"""Correcao OFFLINE e GRATUITA da soletracao -- sem IA, sem API, sem custo.

Alternativa gratuita ao `CorretorLLM`. Usa um dicionario de frequencia do
portugues (`pyspellchecker`) e corrige cada palavra pela distancia de edicao:
a palavra do dicionario mais provavel a poucas letras de distancia.

    "CAURO" -> (R que virou U, 1 letra de distancia) -> "carro" (ou "couro"!)

CUMPRE A MESMA INTERFACE `Corretor` que o `CorretorLLM`. Por isso a demo e o
backend podem usar um ou outro sem mudar nada em volta (ADR-019).

A LIMITACAO HONESTA -- e a razao de o LLM existir:

Este corretor nao tem CONTEXTO. "CAURO" esta a uma letra de "carro" E de
"couro"; sem a frase ao redor, ele escolhe pela palavra mais frequente e pode
errar. Tambem nao poe acento nem pontuacao nem entende gramatica. Ele conserta
o obvio de graca; o contexto e o que voce paga no LLM (ADR-020).
"""

from __future__ import annotations

from collections.abc import Callable

from libras.texto.corretor import ErroCorretor

# Uma funcao que corrige UMA palavra. Injetavel para testar sem o dicionario.
CorrigirPalavra = Callable[[str], str]


def pyspellchecker_disponivel() -> bool:
    try:
        import spellchecker  # noqa: F401
    except ImportError:
        return False
    return True


class CorretorLocal:
    """Corrige texto soletrado palavra por palavra, offline.

    corretor = CorretorLocal()                          # dicionario real
    corretor = CorretorLocal(corrigir_palavra=fake)     # nos testes

    corretor.corrigir("CAURO OI")  # -> "carro oi" (ou similar)
    """

    def __init__(
        self,
        corrigir_palavra: CorrigirPalavra | None = None,
        *,
        distancia: int = 2,
    ) -> None:
        self._corrigir_palavra = corrigir_palavra
        # distancia de edicao maxima que o corretor considera. 2 pega trocas +
        # insercoes/remocoes; acima disso vira chute.
        self.distancia = distancia

    def _obter_corretor(self) -> CorrigirPalavra:
        if self._corrigir_palavra is not None:
            return self._corrigir_palavra

        try:
            from spellchecker import SpellChecker
        except ImportError as exc:
            raise ErroCorretor(
                "o pacote 'pyspellchecker' nao esta instalado.\n"
                "Rode:  .venv\\Scripts\\python.exe -m pip install pyspellchecker"
            ) from exc

        sc = SpellChecker(language="pt", distance=self.distancia)

        def corrigir(palavra: str) -> str:
            baixa = palavra.lower()
            if baixa in sc:  # ja e uma palavra valida
                return baixa
            candidata = sc.correction(baixa)
            # Se o dicionario nao acha nada plausivel, MANTEM a soletracao
            # original em maiusculas -- nao inventamos palavra (mesma regra do
            # prompt do LLM). Datilologia serve para nomes proprios que nao
            # estao em dicionario nenhum.
            return candidata if candidata else palavra.upper()

        self._corrigir_palavra = corrigir
        return corrigir

    def corrigir(self, texto_bruto: str, contexto: str | None = None) -> str:
        # `contexto` e IGNORADO de proposito: a versao offline nao o usa. O
        # parametro existe so para cumprir a interface Corretor -- e a assinatura
        # e identica a do LLM, entao trocar um pelo outro nao quebra ninguem.
        corrigir_palavra = self._obter_corretor()
        palavras = texto_bruto.strip().split()
        return " ".join(corrigir_palavra(p) for p in palavras)


class CorretorPassthrough:
    """Ultimo recurso: devolve o texto soletrado cru, sem corrigir.

    Existe para o sistema nunca ficar sem legenda: se nao ha nem LLM nem
    dicionario, a legenda vira a propria soletracao ("C A S A"). Feio, mas
    funciona -- e sempre melhor mostrar algo do que quebrar.
    """

    def corrigir(self, texto_bruto: str, contexto: str | None = None) -> str:
        return texto_bruto.strip()
