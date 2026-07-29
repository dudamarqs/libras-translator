"""Montagem de mensagem: sequencia de letras reconhecidas -> texto soletrado.

Esta e a parte MECANICA e DETERMINISTICA da legenda (Etapa 10). Ela nao usa IA:
so acumula as letras que o reconhecedor confirma e as agrupa em palavras.

Por que ela e um modulo separado do corretor por IA (`corretor.py`):

  - E logica pura -- testavel sem camera, sem internet, sem chave de API e sem
    gastar um centavo. Voce consegue verificar que "C","A","S","A" vira "CASA"
    num teste de milissegundos.
  - Separa duas responsabilidades diferentes: MONTAR o texto (previsivel) e
    CORRIGIR o contexto (probabilistico, pago). O corretor recebe o texto que
    este modulo produz. Se um dia trocarmos o LLM, este modulo nao muda.

O reconhecedor (scripts/reconhecer.py) ja faz o *debounce*: ele confirma uma
letra UMA vez, quando a mao fica estavel. Entao cada chamada a `letra()` aqui e
uma letra ja confirmada -- este modulo nao precisa lidar com os 30 frames por
segundo, so com o fluxo limpo de letras.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MontadorDeMensagem:
    """Acumula letras confirmadas em palavras e frase.

    Modelo mental:
      - `letra(c)`          -> a mao soletrou mais uma letra da palavra atual.
      - `fim_de_palavra()`  -> houve uma PAUSA (mao saiu do quadro): fecha a
                               palavra atual e comeca uma nova. Em datilologia,
                               tirar a mao entre palavras e o "espaco".
      - `apagar()`          -> desfaz a ultima letra (ou a ultima palavra, se a
                               palavra atual estiver vazia). Erros acontecem;
                               desfazer nao pode exigir recomecar tudo.
      - `texto_bruto()`     -> o texto soletrado ate agora, ex.: "CAURO OI".
                               "bruto" porque ainda NAO passou pela IA -- pode
                               ter erros de reconhecimento (o R que virou U).
    """

    _palavra_atual: list[str] = field(default_factory=list)
    _palavras: list[str] = field(default_factory=list)

    def letra(self, caractere: str) -> None:
        # Uma letra por chamada. Nao validamos contra o alfabeto aqui de
        # proposito: o montador confia no reconhecedor. Validar duas vezes so
        # espalha a mesma regra por dois lugares (e a divergencia entre elas
        # vira bug). Uma responsabilidade, um lugar.
        self._palavra_atual.append(caractere)

    def fim_de_palavra(self) -> None:
        # Idempotente: pausas repetidas (a mao fica fora do quadro varios
        # frames) NAO criam palavras vazias nem espacos duplos. Sem isto, um
        # segundo sem mao viraria "OI      TUDO" com espacos a mais.
        if self._palavra_atual:
            self._palavras.append("".join(self._palavra_atual))
            self._palavra_atual = []

    def apagar(self) -> None:
        """Desfaz a ultima acao: tira uma letra, ou reabre a palavra anterior."""
        if self._palavra_atual:
            self._palavra_atual.pop()
        elif self._palavras:
            # Palavra atual vazia -> "descarta o espaco" e volta a editar a
            # ultima palavra fechada. E o que a intuicao de um backspace espera.
            self._palavra_atual = list(self._palavras.pop())

    def limpar(self) -> None:
        self._palavra_atual = []
        self._palavras = []

    def texto_bruto(self) -> str:
        """Todas as palavras + a que esta sendo soletrada, separadas por espaco."""
        partes = list(self._palavras)
        if self._palavra_atual:
            partes.append("".join(self._palavra_atual))
        return " ".join(partes)

    @property
    def vazio(self) -> bool:
        return not self._palavras and not self._palavra_atual

    @property
    def palavra_em_construcao(self) -> str:
        """A palavra sendo soletrada agora (para mostrar 'ao vivo' na tela)."""
        return "".join(self._palavra_atual)
