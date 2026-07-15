"""Testes do registro de sinais."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from libras.registry import carregar_registro, registro


class TestRegistroPadrao:
    def test_carrega_o_sinais_yaml_do_projeto(self) -> None:
        reg = registro()
        assert len(reg.sinais) > 0
        assert "A" in reg.nomes

    def test_estaticos_e_dinamicos_particionam_o_registro(self) -> None:
        reg = registro()
        assert len(reg.estaticos()) + len(reg.dinamicos()) == len(reg.sinais)
        assert all(s.estatico for s in reg.estaticos())
        assert all(not s.estatico for s in reg.dinamicos())

    def test_letras_com_movimento_estao_marcadas_dinamicas(self) -> None:
        # Guarda a decisao honesta: J/K/etc tem movimento e NAO devem entrar na
        # coleta estatica. Se alguem marca-las como estaticas por engano, a
        # coleta da fase atual tentaria grava-las em 1 frame.
        reg = registro()
        dinamicas = {s.nome for s in reg.dinamicos()}
        assert {"J", "K", "Z"} <= dinamicas

    def test_get_desconhecido_falha_claro(self) -> None:
        with pytest.raises(KeyError, match="registro"):
            registro().get("9")


class TestValidacao:
    def _escrever(self, tmp_path: Path, conteudo: str) -> Path:
        arq = tmp_path / "sinais.yaml"
        arq.write_text(textwrap.dedent(conteudo), encoding="utf-8")
        return arq

    def test_tipo_invalido_e_pego_na_carga(self, tmp_path: Path) -> None:
        # Um typo ('estatco') vira erro AGORA, e nao um KeyError no meio do
        # treino tres horas depois.
        arq = self._escrever(
            tmp_path,
            """
            sinais:
              A: {tipo: estatco, categoria: letra, maos: 1}
            """,
        )
        with pytest.raises(ValueError, match="tipo"):
            carregar_registro(arq)

    def test_numero_de_maos_invalido_e_pego(self, tmp_path: Path) -> None:
        arq = self._escrever(
            tmp_path,
            """
            sinais:
              A: {tipo: estatico, categoria: letra, maos: 3}
            """,
        )
        with pytest.raises(ValueError, match="maos"):
            carregar_registro(arq)

    def test_yaml_sem_chave_sinais_falha(self, tmp_path: Path) -> None:
        arq = self._escrever(tmp_path, "outra_coisa: 1\n")
        with pytest.raises(ValueError, match="sinais"):
            carregar_registro(arq)
