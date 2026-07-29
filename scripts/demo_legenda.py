"""Demo da camada de legenda: soletracao ruidosa -> legenda (Etapa 10).

    .venv\\Scripts\\python.exe scripts\\demo_legenda.py

Nao precisa de camera. Voce ve a soletracao crua (como o reconhecedor cospe,
com os erros de R/U, T/F, sem acento) corrigida por dois caminhos:

  - OFFLINE (gratis): dicionario do portugues + distancia de edicao. Corrige
    palavra por palavra, SEM contexto. Roda sem chave nenhuma.
  - CLAUDE (pago): usa o contexto da frase para desambiguar, poe acento e
    pontuacao. So roda se houver o SDK + ANTHROPIC_API_KEY.

O contraste e a licao: onde o gratis basta, e onde o contexto do LLM importa.
"""

from __future__ import annotations

from libras.texto import CorretorLLM, CorretorLocal, ErroCorretor
from libras.texto.corretor import sdk_disponivel
from libras.texto.corretor_local import pyspellchecker_disponivel

# Soletracao RUIDOSA (com os erros medidos no reconhecedor) + contexto que a
# legenda ja tinha. O contexto e o que so o LLM usa.
EXEMPLOS = [
    ("CAURO", "Eu comprei um"),  # R->U, ambiguo: carro? couro? (contexto decide)
    ("OI TUDO BM", None),  # sem acento/pontuacao
    ("MEZA", "A comida esta na"),  # Z->S, sem acento
    ("OBRIGADA", None),  # ja correto: so acento
]


def _linha(rotulo: str, corretor, usa_contexto: bool) -> None:
    print(f"\n  {rotulo}")
    print("  " + "-" * 58)
    for bruto, contexto in EXEMPLOS:
        try:
            ctx = contexto if usa_contexto else None
            saida = corretor.corrigir(bruto, contexto=ctx)
        except ErroCorretor as exc:
            print(f"  ERRO: {exc}")
            return
        marca = f"  (contexto: {contexto})" if usa_contexto and contexto else ""
        print(f"  {bruto:<14} -> {saida}{marca}")


def main() -> int:
    print("\n" + "=" * 62)
    print("DEMO -- camada de legenda (Etapa 10)")
    print("=" * 62)

    if pyspellchecker_disponivel():
        _linha("OFFLINE (gratis, sem contexto)", CorretorLocal(), usa_contexto=False)
    else:
        print("\n  [offline indisponivel] pip install pyspellchecker")

    if sdk_disponivel():
        try:
            _linha("CLAUDE (usa contexto, poe acento/pontuacao)", CorretorLLM(), usa_contexto=True)
        except ErroCorretor as exc:
            print(f"\n  [Claude indisponivel] {exc}")
    else:
        print(
            "\n  [Claude indisponivel] para comparar com o LLM:\n"
            "    pip install anthropic  +  defina ANTHROPIC_API_KEY"
        )

    print(
        "\nRepare em 'CAURO': o offline chuta pela palavra mais comum e pode ir\n"
        "para 'couro'; o Claude usa 'Eu comprei um' e acerta 'carro'. Contexto e\n"
        "exatamente o que voce paga no LLM -- e o que o reconhecedor sozinho nao tem.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
