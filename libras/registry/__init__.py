"""Registro de sinais: o catalogo declarativo, carregado do sinais.yaml.

Por que um registro, e nao constantes espalhadas pelo codigo:

O requisito e "adicionar sinais sem reescrever o sistema". Se a lista de sinais
morasse dentro do coletor, do treino e da inferencia (tres copias), adicionar
uma letra exigiria mexer nos tres -- e um dia divergiriam. Com o registro, a
lista vive num lugar so (o YAML), e os tres leem dela.

Este e o mesmo principio do `libras/` como fonte unica (ADR-004), aplicado aos
METADADOS em vez do codigo.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

_ARQUIVO_PADRAO = Path(__file__).resolve().parent / "sinais.yaml"

TIPOS_VALIDOS = {"estatico", "dinamico"}


@dataclass(slots=True, frozen=True)
class Sinal:
    nome: str
    tipo: str  # "estatico" | "dinamico"
    categoria: str  # "letra" | "palavra"
    maos: int

    @property
    def estatico(self) -> bool:
        return self.tipo == "estatico"


@dataclass(slots=True, frozen=True)
class Registro:
    """Todos os sinais conhecidos pelo sistema."""

    sinais: tuple[Sinal, ...]

    def get(self, nome: str) -> Sinal:
        for s in self.sinais:
            if s.nome == nome:
                return s
        raise KeyError(f"sinal '{nome}' nao esta no registro (sinais.yaml)")

    @property
    def nomes(self) -> tuple[str, ...]:
        return tuple(s.nome for s in self.sinais)

    def estaticos(self) -> tuple[Sinal, ...]:
        """Sinais coletaveis/treinaveis na fase atual (1 frame)."""
        return tuple(s for s in self.sinais if s.estatico)

    def dinamicos(self) -> tuple[Sinal, ...]:
        return tuple(s for s in self.sinais if not s.estatico)


def _validar(nome: str, dados: dict) -> Sinal:
    tipo = dados.get("tipo")
    if tipo not in TIPOS_VALIDOS:
        raise ValueError(f"sinal '{nome}': tipo '{tipo}' invalido (esperado {TIPOS_VALIDOS})")
    maos = dados.get("maos", 1)
    if maos not in (1, 2):
        raise ValueError(f"sinal '{nome}': maos={maos} invalido (esperado 1 ou 2)")
    return Sinal(
        nome=nome,
        tipo=tipo,
        categoria=dados.get("categoria", "letra"),
        maos=int(maos),
    )


def carregar_registro(caminho: Path | None = None) -> Registro:
    """Le e VALIDA o sinais.yaml.

    Validamos na carga, cedo e alto: um typo no YAML (tipo: 'estatco') vira um
    erro claro AGORA, e nao um KeyError obscuro no meio do treino, tres horas
    depois de comecar a coletar.
    """
    caminho = caminho or _ARQUIVO_PADRAO
    with caminho.open(encoding="utf-8") as f:
        bruto = yaml.safe_load(f)

    if not bruto or "sinais" not in bruto:
        raise ValueError(f"{caminho} nao tem a chave 'sinais'")

    sinais = tuple(_validar(nome, dados) for nome, dados in bruto["sinais"].items())
    if not sinais:
        raise ValueError(f"{caminho} nao lista nenhum sinal")

    return Registro(sinais=sinais)


@lru_cache(maxsize=1)
def registro() -> Registro:
    """O registro padrao, carregado uma vez e memoizado."""
    return carregar_registro()
