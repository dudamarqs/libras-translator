"""Testes do armazenamento do dataset.

O foco: o round-trip (salvar -> carregar devolve o mesmo?) e a rastreabilidade
dos GRUPOS (cada amostra sabe de que sessao veio?). O grupo e o que torna o
split treino/teste honesto -- se ele se perde no salvamento/carregamento, o
data leakage volta sem ninguem perceber.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from libras.data.dataset import (
    Sessao,
    carregar_dataset,
    carregar_sessao,
    salvar_sessao,
)
from libras.vision.hands import N_LANDMARKS


def sessao_falsa(session_id: str, rotulos: list[str]) -> Sessao:
    n = len(rotulos)
    rng = np.random.default_rng(abs(hash(session_id)) % (2**32))
    return Sessao(
        session_id=session_id,
        world=rng.normal(0, 0.05, (n, N_LANDMARKS, 3)).astype(np.float32),
        imagem=rng.random((n, N_LANDMARKS, 3)).astype(np.float32),
        lado=np.array(["Right"] * n),
        rotulo=np.array(rotulos),
    )


class TestSessao:
    def test_arrays_de_tamanhos_diferentes_falham(self) -> None:
        with pytest.raises(ValueError, match="comprimentos"):
            Sessao(
                session_id="s",
                world=np.zeros((3, N_LANDMARKS, 3), np.float32),
                imagem=np.zeros((3, N_LANDMARKS, 3), np.float32),
                lado=np.array(["Right"] * 3),
                rotulo=np.array(["A", "B"]),  # so 2!
            )

    def test_world_com_shape_errado_falha(self) -> None:
        with pytest.raises(ValueError, match="world"):
            Sessao(
                session_id="s",
                world=np.zeros((2, 10, 3), np.float32),  # 10 != 21
                imagem=np.zeros((2, 10, 3), np.float32),
                lado=np.array(["Right"] * 2),
                rotulo=np.array(["A", "B"]),
            )

    def test_contagem_por_rotulo(self) -> None:
        s = sessao_falsa("s1", ["A", "A", "B"])
        assert s.contagem() == {"A": 2, "B": 1}


class TestRoundTrip:
    def test_salvar_e_carregar_preserva_os_dados(self, tmp_path: Path) -> None:
        original = sessao_falsa("sessao_x", ["A", "B", "C", "A"])
        salvar_sessao(original, raiz=tmp_path)

        recarregada = carregar_sessao(tmp_path / "sessao_x")

        np.testing.assert_allclose(recarregada.world, original.world, rtol=1e-6)
        np.testing.assert_array_equal(recarregada.rotulo, original.rotulo)
        np.testing.assert_array_equal(recarregada.lado, original.lado)

    def test_meta_json_e_escrito_com_a_contagem(self, tmp_path: Path) -> None:
        import json

        salvar_sessao(sessao_falsa("sessao_y", ["A", "A", "B"]), raiz=tmp_path)
        meta = json.loads((tmp_path / "sessao_y" / "meta.json").read_text("utf-8"))

        assert meta["n_amostras"] == 3
        assert meta["contagem_por_rotulo"] == {"A": 2, "B": 1}
        assert meta["session_id"] == "sessao_y"


class TestCarregarDataset:
    def test_concatena_varias_sessoes(self, tmp_path: Path) -> None:
        salvar_sessao(sessao_falsa("sessao_1", ["A", "B"]), raiz=tmp_path)
        salvar_sessao(sessao_falsa("sessao_2", ["A", "C", "C"]), raiz=tmp_path)

        ds = carregar_dataset(raiz=tmp_path)

        assert ds.n_amostras == 5
        assert ds.rotulos_unicos == ["A", "B", "C"]
        assert ds.contagem_por_rotulo() == {"A": 2, "B": 1, "C": 2}

    def test_cada_amostra_conhece_a_propria_sessao(self, tmp_path: Path) -> None:
        # ESTE e o teste que protege contra data leakage. Se os grupos se
        # embaralharem, o split por sessao vira split aleatorio disfarçado.
        salvar_sessao(sessao_falsa("sessao_1", ["A", "B"]), raiz=tmp_path)
        salvar_sessao(sessao_falsa("sessao_2", ["C", "D", "E"]), raiz=tmp_path)

        ds = carregar_dataset(raiz=tmp_path)

        assert ds.sessoes_unicas == ["sessao_1", "sessao_2"]
        # As duas primeiras amostras (A, B) vieram da sessao_1; as tres seguintes
        # da sessao_2. A ordem segue a ordenacao das pastas.
        por_sessao = {s: set() for s in ds.sessoes_unicas}
        for rotulo, grupo in zip(ds.rotulo, ds.grupos, strict=True):
            por_sessao[grupo].add(rotulo)
        assert por_sessao["sessao_1"] == {"A", "B"}
        assert por_sessao["sessao_2"] == {"C", "D", "E"}

    def test_pasta_sem_dados_npz_e_ignorada(self, tmp_path: Path) -> None:
        salvar_sessao(sessao_falsa("sessao_boa", ["A"]), raiz=tmp_path)
        (tmp_path / "lixo").mkdir()  # pasta vazia, sem dados.npz
        (tmp_path / "sessao_boa" / "extra.txt").write_text("nada")  # arquivo solto

        ds = carregar_dataset(raiz=tmp_path)
        assert ds.sessoes_unicas == ["sessao_boa"]

    def test_diretorio_vazio_da_erro_util(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="collect"):
            carregar_dataset(raiz=tmp_path)
