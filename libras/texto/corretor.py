"""Correcao de contexto por IA -- a camada que transforma soletracao ruidosa
em portugues natural (Etapa 10).

Esta e a peca que voce pediu desde o comeco: o texto soletrado (com os erros
que o reconhecedor comete -- R que vira U, T que vira F) entra, e a IA devolve
a frase pretendida, corrigida e pontuada.

    "CAURO" -> Claude -> "carro"
    "OI TUDO BM" -> Claude -> "Oi, tudo bem?"

POR QUE UM LLM RESOLVE O QUE NENHUM AJUSTE DE MODELO RESOLVEU:

Nas Etapas 8-9 medimos que o classificador confunde R/U/V e T/F entre sessoes,
e nem a MLP nem a augmentation consertaram (ADR-017, ADR-018). Mas essas
confusoes sao TRIVIAIS de corrigir quando ha uma palavra ao redor: "CAURO" so
pode ser "carro". O gargalo do reconhecedor e a solucao que voce queria
construir sao o mesmo problema visto de dois lados.

DECISOES DE ENGENHARIA:

  - Cliente INJETADO por dependencia. O `import anthropic` e preguicoso (so
    acontece quando um cliente real e construido), entao este modulo importa e
    e testavel mesmo sem o SDK instalado e sem chave de API. Os testes passam
    um cliente FALSO e verificam o prompt/parsing sem gastar um centavo.

  - Roda no BACKEND, nunca no navegador (quando houver frontend): a chave da API
    e um segredo e nao pode ir para o cliente. Por ora vive aqui, em `libras/`.

  - Ao construir esta camada, consultamos a skill `claude-api` para o ID de
    modelo (`claude-opus-5`) e os parametros corretos, em vez de confiar na
    memoria -- as APIs de LLM mudam rapido (mesma licao do ADR-006 do MediaPipe).
"""

from __future__ import annotations

from typing import Protocol

MODELO_PADRAO = "claude-opus-5"


class Corretor(Protocol):
    """A interface que TODO corretor cumpre -- LLM ou offline.

    O montador produz texto; um Corretor o transforma em legenda. Como os dois
    corretores (Claude e o offline gratuito) implementam esta mesma interface, o
    resto do sistema (demo, backend) fala com `Corretor` e nao sabe nem se
    importa qual esta por tras. Trocar um pelo outro nao muda nada em volta.
    """

    def corrigir(self, texto_bruto: str, contexto: str | None = None) -> str: ...


# O prompt de sistema e o CEREBRO desta camada. Ele descreve exatamente a
# tarefa, os erros esperados do reconhecedor, e -- crucial -- manda responder
# SO com o texto corrigido, sem conversa.
PROMPT_SISTEMA = """\
Voce corrige datilologia de Libras (o alfabeto manual) reconhecida por um \
sistema automatico de visao computacional.

O reconhecedor soletra letra por letra e comete erros previsiveis: confunde \
R com U e V (dedos indicador e medio cruzados/juntos/afastados), T com F, e \
M com N. Ele nao poe acentos, nao poe pontuacao e as vezes erra o espaco entre \
palavras.

Sua tarefa: inferir o que a pessoa quis dizer e devolver o texto em portugues \
correto e natural -- corrigindo os erros de reconhecimento, formando palavras \
reais, e adicionando acentos e pontuacao. Use o contexto das frases anteriores \
(quando houver) para desambiguar.

Regras:
- Responda APENAS com o texto corrigido. Nada de explicacoes, aspas ou rotulos.
- Se o texto ja estiver correto, devolva-o como esta (so ajustando acento/pontuacao).
- Se for impossivel adivinhar uma palavra, mantenha a soletracao original em \
maiusculas em vez de inventar.
- Preserve nomes proprios soletrados (datilologia serve justamente para nomes)."""


class ClienteLLM(Protocol):
    """O minimo que o corretor precisa de um cliente Anthropic.

    Um Protocol (tipagem estrutural) em vez de importar o tipo real: qualquer
    objeto com `.messages.create(...)` serve -- inclusive o cliente FALSO dos
    testes. E o que desacopla este modulo do SDK.
    """

    messages: object


class ErroCorretor(RuntimeError):
    """Falha ao chamar o LLM (sem SDK, sem chave, rede fora, etc.)."""


def sdk_disponivel() -> bool:
    """O pacote `anthropic` esta instalado?"""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


class CorretorLLM:
    """Corrige texto soletrado usando o Claude.

    corretor = CorretorLLM()                       # cliente real (precisa de chave)
    corretor = CorretorLLM(cliente=cliente_falso)  # nos testes

    legenda = corretor.corrigir("CAURO", contexto="Eu tenho um")
    """

    def __init__(
        self,
        cliente: ClienteLLM | None = None,
        *,
        modelo: str = MODELO_PADRAO,
        max_tokens: int = 256,
    ) -> None:
        self._cliente = cliente
        self.modelo = modelo
        # 256 tokens: a saida e uma frase curta. Legenda nao e redacao -- limitar
        # a saida corta custo e latencia, que importam num sistema em tempo real.
        self.max_tokens = max_tokens

    def _obter_cliente(self) -> ClienteLLM:
        if self._cliente is not None:
            return self._cliente
        # Construcao preguicosa do cliente real. So chega aqui em producao, nunca
        # nos testes (que injetam o cliente). Por isso o import fica aqui dentro.
        try:
            import anthropic
        except ImportError as exc:
            raise ErroCorretor(
                "o pacote 'anthropic' nao esta instalado.\n"
                "Rode:  .venv\\Scripts\\python.exe -m pip install anthropic"
            ) from exc
        try:
            # Anthropic() resolve a credencial do ambiente (ANTHROPIC_API_KEY,
            # ou um perfil `ant auth login`). Nao passamos a chave no codigo --
            # segredo em codigo e o classico vazamento no git.
            self._cliente = anthropic.Anthropic()
        except Exception as exc:  # noqa: BLE001
            raise ErroCorretor(f"nao consegui criar o cliente Anthropic: {exc}") from exc
        return self._cliente

    def corrigir(self, texto_bruto: str, contexto: str | None = None) -> str:
        """Texto soletrado (cru) -> portugues corrigido.

        `contexto`: as frases ja confirmadas antes desta, para desambiguar.
        Levanta ErroCorretor se a chamada ao LLM falhar.
        """
        texto_bruto = texto_bruto.strip()
        if not texto_bruto:
            return ""

        cliente = self._obter_cliente()

        conteudo = f"Soletrado: {texto_bruto}"
        if contexto:
            conteudo = f"Contexto anterior: {contexto}\n{conteudo}"

        try:
            resposta = cliente.messages.create(
                model=self.modelo,
                max_tokens=self.max_tokens,
                # effort "low": correcao ortografica com contexto e uma tarefa
                # simples. Effort baixo = mais rapido e barato, o que importa
                # numa legenda ao vivo. (thinking fica no adaptativo padrao.)
                output_config={"effort": "low"},
                system=PROMPT_SISTEMA,
                messages=[{"role": "user", "content": conteudo}],
            )
        except Exception as exc:  # noqa: BLE001
            raise ErroCorretor(f"falha ao chamar o LLM: {exc}") from exc

        return _extrair_texto(resposta)


def _extrair_texto(resposta: object) -> str:
    """Junta os blocos de texto da resposta do Claude.

    A resposta e uma LISTA de blocos (pode haver blocos de 'thinking' junto dos
    de 'text'). Filtramos por type == 'text' -- ler `content[0].text` as cegas
    quebraria se o primeiro bloco fosse 'thinking' (o erro classico da API).
    """
    blocos = getattr(resposta, "content", None)
    if not blocos:
        return ""
    partes = [getattr(b, "text", "") for b in blocos if getattr(b, "type", None) == "text"]
    return "".join(partes).strip()
