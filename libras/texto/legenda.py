"""Legenda AO VIVO -- liga o reconhecedor (rapido) a correcao de texto (lenta).

Esta e a peca que junta as duas metades do projeto (Etapa 11). Ate aqui elas
existiam separadas: `scripts/reconhecer.py` cuspia letras cruas na tela, e
`libras/texto/` (montador + corretor) so rodava numa demo sem camera.

O PROBLEMA QUE ESTE MODULO EXISTE PARA RESOLVER -- tempo.

O loop da camera roda a 30 FPS: ele tem 33 ms para fazer TUDO num frame. Uma
correcao demora muito mais que isso:

    CorretorLocal (dicionario)  ~  50 ms      -> ja perde 1-2 frames
    CorretorLLM (Claude, rede)  ~  1-3 s      -> congela 30-90 frames

Chamar `corrigir()` dentro do loop travaria o video a cada palavra. E o erro
classico de misturar trabalho de I/O com um laco de tempo real. A solucao nao e
"deixar o corretor mais rapido" (a rede nao obedece): e TIRAR a correcao do
caminho do frame. Uma thread trabalhadora corrige em segundo plano; o loop so
LE o ultimo resultado pronto e desenha. O loop nunca bloqueia -- no maximo
mostra uma legenda de meio segundo atras, que e o comportamento de qualquer
legenda ao vivo de TV.

DECISOES QUE ESSA ESCOLHA IMPOE:

  - **Um pedido de cada vez, o ultimo vence.** Nao ha fila. Se voce soletra mais
    uma palavra enquanto o Claude ainda pensa na anterior, o pedido antigo e
    substituido. Numa legenda, uma correcao que ficou obsoleta durante a viagem
    nao tem valor nenhum -- descartar e o comportamento certo, nao uma perda.

  - **Geracao (`_geracao`).** Se voce limpa a tela enquanto uma correcao esta em
    voo, o resultado nao pode ressuscitar o texto apagado. Cada pedido carrega o
    numero da geracao em que nasceu; ao voltar, se a geracao mudou, o resultado
    e jogado fora. Sem isso existiria um bug raro e confuso: a legenda "voltando
    do nada" depois do usuario apertar limpar.

  - **Fallback.** Se o LLM falhar (rede caiu, sem credito), cai no corretor
    offline em vez de ficar sem legenda. Degradar e melhor que quebrar.

  - **So pede correcao no FIM DE PALAVRA**, nao a cada letra. Cada pedido ao
    Claude custa dinheiro e latencia; corrigir "C", "CA", "CAS", "CASA" e gastar
    quatro chamadas para chegar no mesmo lugar. A pausa (mao fora do quadro) e a
    fronteira natural -- e de graca.

QUEM CHAMA O QUE (importante para nao precisar de lock no montador):
    thread da camera  -> letra/fim_de_palavra/apagar/limpar/nova_frase e as
                         propriedades de leitura.
    thread de fundo   -> so o corretor e os campos protegidos por `_cond`.
O montador e tocado APENAS pela thread da camera, entao nao precisa de lock.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from libras.texto.corretor import Corretor, ErroCorretor
from libras.texto.montador import MontadorDeMensagem

# Quantas frases ja fechadas seguem servindo de contexto para a proxima.
# Contexto ajuda o LLM a desambiguar, mas mandar a sessao inteira a cada pedido
# custa tokens que crescem sem limite. Tres frases e memoria de conversa curta.
FRASES_DE_CONTEXTO = 3


class LegendaAoVivo:
    """Montador + corretor + uma thread, com a interface que o loop precisa.

        with LegendaAoVivo(criar_corretor()) as legenda:
            legenda.letra("C")            # o reconhecedor confirmou um "C"
            legenda.fim_de_palavra()      # a mao saiu do quadro -> corrige
            print(legenda.texto)          # o que desenhar na tela, agora

    Nenhum metodo bloqueia (exceto `aguardar` e `fechar`, de proposito).
    """

    def __init__(
        self,
        corretor: Corretor,
        *,
        fallback: Corretor | None = None,
        montador: MontadorDeMensagem | None = None,
    ) -> None:
        self._corretor = corretor
        self._fallback = fallback
        self._montador = montador if montador is not None else MontadorDeMensagem()

        # Tudo daqui para baixo e compartilhado entre as duas threads e so pode
        # ser lido/escrito com `_cond` em maos.
        self._cond = threading.Condition()
        self._pedido: str | None = None  # texto esperando correcao (slot de 1)
        self._em_voo = False  # o trabalhador esta dentro do corretor agora?
        self._geracao = 0  # invalida resultados de um texto que ja morreu
        self._legenda = ""  # ultimo resultado corrigido
        self._erro: str | None = None
        self._frases: deque[str] = deque(maxlen=FRASES_DE_CONTEXTO)
        self._parar = False

        self._trabalhador = threading.Thread(
            target=self._laco, name="legenda-corretor", daemon=True
        )
        self._trabalhador.start()

    # -- entrada: chamado pelo loop da camera --------------------------------

    def letra(self, caractere: str) -> None:
        """O reconhecedor confirmou mais uma letra. NAO dispara correcao."""
        self._montador.letra(caractere)

    def fim_de_palavra(self) -> None:
        """Houve pausa (mao fora do quadro): fecha a palavra e pede correcao.

        Ignora pausas repetidas. Isto importa mais do que parece: o loop chama
        isto a CADA frame em que nao ha mao -- 30 vezes por segundo. Sem esta
        guarda, um segundo de mao fora do quadro viraria 30 chamadas ao Claude.
        """
        if not self._montador.palavra_em_construcao:
            return
        self._montador.fim_de_palavra()
        self.solicitar_correcao()

    def apagar(self) -> None:
        """Desfaz a ultima letra (ou reabre a palavra anterior)."""
        self._montador.apagar()

    def limpar(self) -> None:
        """Zera tudo. Uma correcao em voo agora e descartada quando voltar."""
        self._montador.limpar()
        with self._cond:
            self._geracao += 1
            self._pedido = None
            self._legenda = ""
            self._erro = None
            self._cond.notify_all()

    def nova_frase(self) -> None:
        """Fecha a frase atual: ela vira CONTEXTO e o montador recomeca vazio.

        E o "ponto final" da legenda. Serve a dois propositos ao mesmo tempo:
        da ao LLM a memoria do que ja foi dito, e impede que o texto enviado
        cresca para sempre numa sessao longa (cada pedido pagaria a sessao
        inteira em tokens).
        """
        self._montador.fim_de_palavra()
        bruto = self._montador.texto_bruto()
        if not bruto:
            return
        with self._cond:
            # Preferimos arquivar a versao CORRIGIDA; se ela ainda nao chegou,
            # o bruto serve -- contexto aproximado e melhor que contexto nenhum.
            self._frases.append(self._legenda or bruto)
            self._geracao += 1
            self._pedido = None
            self._legenda = ""
            self._erro = None
            self._cond.notify_all()
        self._montador.limpar()

    def solicitar_correcao(self) -> None:
        """Poe o texto atual na fila de 1 lugar. Volta na hora, nao bloqueia."""
        bruto = self._montador.texto_bruto()
        if not bruto:
            return
        with self._cond:
            self._pedido = bruto
            self._cond.notify_all()

    # -- saida: chamado pelo loop da camera para desenhar --------------------

    @property
    def texto_bruto(self) -> str:
        """A soletracao crua, como o reconhecedor produziu."""
        return self._montador.texto_bruto()

    @property
    def palavra_em_construcao(self) -> str:
        return self._montador.palavra_em_construcao

    @property
    def legenda(self) -> str:
        """A ultima correcao pronta ("" enquanto a primeira nao chega)."""
        with self._cond:
            return self._legenda

    @property
    def texto(self) -> str:
        """O que mostrar na tela: a legenda corrigida, ou o bruto enquanto nao ha.

        Sempre devolve alguma coisa. Uma legenda que fica em branco esperando a
        rede parece um sistema quebrado -- e melhor mostrar a soletracao crua e
        deixar a correcao substitui-la quando chegar.
        """
        with self._cond:
            if self._legenda:
                return self._legenda
        return self._montador.texto_bruto()

    @property
    def corrigindo(self) -> bool:
        """Ha um pedido esperando ou em voo? (para piscar um "..." na tela)"""
        with self._cond:
            return self._pedido is not None or self._em_voo

    @property
    def erro(self) -> str | None:
        """Mensagem da ultima falha de correcao, ou None se a ultima deu certo."""
        with self._cond:
            return self._erro

    @property
    def contexto(self) -> str | None:
        """As frases ja fechadas, que vao junto no proximo pedido."""
        with self._cond:
            return " ".join(self._frases) if self._frases else None

    # -- ciclo de vida -------------------------------------------------------

    def aguardar(self, timeout: float = 10.0) -> bool:
        """Bloqueia ate nao haver correcao pendente. True se esvaziou a tempo.

        O loop da camera NAO deve chamar isto (bloquear e justamente o que
        estamos evitando). Existe para os testes e para o encerramento, onde
        esperar o ultimo resultado e o certo.
        """
        limite = time.monotonic() + timeout
        with self._cond:
            while self._pedido is not None or self._em_voo:
                restante = limite - time.monotonic()
                if restante <= 0:
                    return False
                self._cond.wait(restante)
        return True

    def fechar(self) -> None:
        """Encerra a thread. Idempotente -- pode chamar duas vezes sem medo."""
        with self._cond:
            if self._parar:
                return
            self._parar = True
            self._cond.notify_all()
        # A thread e daemon: mesmo que ela esteja presa numa chamada de rede
        # lenta, o processo consegue sair. O join com timeout e so a tentativa
        # educada de esperar o encerramento limpo.
        self._trabalhador.join(timeout=2.0)

    def __enter__(self) -> LegendaAoVivo:
        return self

    def __exit__(self, *_excecao: object) -> None:
        self.fechar()

    # -- a thread de fundo ---------------------------------------------------

    def _laco(self) -> None:
        while True:
            with self._cond:
                while self._pedido is None and not self._parar:
                    self._cond.wait()
                if self._parar:
                    return
                texto = self._pedido
                self._pedido = None
                self._em_voo = True
                geracao = self._geracao
                contexto = " ".join(self._frases) if self._frases else None

            # FORA do lock: a chamada lenta. Segurar o lock aqui faria o loop da
            # camera travar em `legenda.texto` -- exatamente o que este modulo
            # existe para impedir.
            resultado: str | None = None
            erro: str | None = None
            try:
                resultado = self._corrigir(texto, contexto)
            except Exception as exc:  # noqa: BLE001 -- a thread nao pode morrer
                erro = str(exc)

            with self._cond:
                # Chegou tarde demais? `limpar()`/`nova_frase()` mudaram a
                # geracao, entao este texto ja nao existe mais na tela.
                if geracao == self._geracao:
                    if resultado is not None:
                        self._legenda = resultado
                    self._erro = erro
                self._em_voo = False
                self._cond.notify_all()

    def _corrigir(self, texto: str, contexto: str | None) -> str:
        try:
            return self._corretor.corrigir(texto, contexto=contexto)
        except ErroCorretor:
            if self._fallback is None:
                raise
            # O LLM falhou (rede, credito, chave). O offline nao tem contexto,
            # mas conserta o obvio -- e a legenda continua aparecendo.
            return self._fallback.corrigir(texto, contexto=contexto)
