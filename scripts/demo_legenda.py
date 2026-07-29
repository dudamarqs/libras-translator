"""Demo da camada de IA: soletracao ruidosa -> legenda corrigida (Etapa 10).

    .venv\\Scripts\\python.exe scripts\\demo_legenda.py

Nao precisa de camera. Voce digita o texto SOLETRADO (como o reconhecedor
cospe, com os erros de R/U, T/F, sem acento), e o Claude devolve a legenda.
E a prova de conceito do produto: mostra que as confusoes que o classificador
NAO consegue resolver (ADR-017/018) somem quando ha contexto.

Precisa de:
  1. o SDK:   .venv\\Scripts\\python.exe -m pip install anthropic
  2. a chave: definir ANTHROPIC_API_KEY no ambiente, ou `ant auth login`

Sem esses dois, o script explica o que falta e sai -- ele nao inventa resposta.
"""

from __future__ import annotations

from libras.texto.corretor import CorretorLLM, ErroCorretor, sdk_disponivel

# Exemplos de soletracao RUIDOSA (com os erros que medimos no reconhecedor) e o
# contexto que a legenda ja tinha. O contexto e o que desambigua.
EXEMPLOS = [
    ("CAURO", "Eu comprei um"),  # R->U: "carro"
    ("OI TUDO BM", None),  # sem acento/pontuacao: "Oi, tudo bem?"
    ("MEZA", "A comida esta na"),  # Z->S, sem acento: "mesa"
    ("OBRIGADA", None),  # ja correto: so acento
]


def main() -> int:
    print("\n" + "=" * 60)
    print("DEMO -- camada de IA da legenda (Etapa 10)")
    print("=" * 60)

    if not sdk_disponivel():
        print(
            "\nO pacote 'anthropic' nao esta instalado. Para rodar a demo ao vivo:\n"
            "  .venv\\Scripts\\python.exe -m pip install anthropic\n"
            "e defina ANTHROPIC_API_KEY (ou rode `ant auth login`).\n\n"
            "Mesmo sem isso, a camada esta construida e testada (tests/test_corretor.py\n"
            "usa um cliente falso). So a chamada AO VIVO e que precisa da chave.\n"
        )
        return 1

    corretor = CorretorLLM()
    print(
        f"\nModelo: {corretor.modelo}   (troque para claude-haiku-4-5 se quiser\n"
        "mais barato/rapido numa legenda de producao)\n"
    )

    print(f"{'soletrado (cru)':<18} {'contexto':<22} -> legenda")
    print("-" * 70)
    for bruto, contexto in EXEMPLOS:
        try:
            legenda = corretor.corrigir(bruto, contexto=contexto)
        except ErroCorretor as exc:
            print(f"\nERRO: {exc}\n")
            return 1
        ctx = contexto or "(nenhum)"
        print(f"{bruto:<18} {ctx:<22} -> {legenda}")

    print("\nRepare: 'CAURO' (R que o reconhecedor leu como U) virou 'carro' por")
    print("causa do contexto. Nenhum ajuste de modelo resolveu isso; a IA resolve\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
