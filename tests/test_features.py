"""Testes da normalizacao de features.

Estes sao os testes mais importantes do projeto ate agora.

`normalizar_mao` e `vetor_features` sao chamadas TANTO pelo script de coleta
QUANTO pelo servidor de inferencia. Um bug aqui nao levanta excecao, nao aparece
em log nenhum: ele so faz o modelo receber, em producao, features com
distribuicao diferente da que viu no treino. O modelo simplesmente erra.
Isso e o `training/serving skew` (ADR-004), e e por isso que aqui a gente
testa as PROPRIEDADES MATEMATICAS, e nao so "roda sem estourar".
"""

from __future__ import annotations

import numpy as np
import pytest

from libras.vision.features import (
    MEDIO_MCP,
    N_FEATURES_MAO,
    N_FEATURES_TOTAL,
    PULSO,
    escala_da_mao,
    normalizar_mao,
    vetor_features,
    vetor_uma_mao,
)
from libras.vision.hands import N_LANDMARKS, Mao, ResultadoMaos


def mao_sintetica(*, deslocamento=(0.0, 0.0, 0.0), escala: float = 1.0) -> np.ndarray:
    """Uma 'mao' fixa, que podemos mover e redimensionar a vontade."""
    rng = np.random.default_rng(42)
    base = rng.normal(0, 0.05, size=(N_LANDMARKS, 3)).astype(np.float32)
    base[PULSO] = [0.0, 0.08, 0.0]  # pulso
    base[MEDIO_MCP] = [0.0, 0.0, 0.0]  # base do medio -> regua de 8 cm
    return (base * escala + np.array(deslocamento, dtype=np.float32)).astype(np.float32)


def fazer_mao(lado: str, world: np.ndarray) -> Mao:
    return Mao(
        lado=lado,
        lado_bruto=lado,
        confianca_lado=0.99,
        landmarks=np.zeros((N_LANDMARKS, 3), dtype=np.float32),
        world=world,
    )


class TestEscalaDaMao:
    def test_e_a_distancia_pulso_ate_base_do_medio(self) -> None:
        world = mao_sintetica()
        esperado = np.linalg.norm(world[MEDIO_MCP] - world[PULSO])
        assert escala_da_mao(world) == pytest.approx(esperado)

    def test_dobra_quando_a_mao_dobra_de_tamanho(self) -> None:
        pequena = escala_da_mao(mao_sintetica(escala=1.0))
        grande = escala_da_mao(mao_sintetica(escala=2.0))
        assert grande == pytest.approx(2 * pequena)


class TestNormalizarMao:
    def test_shape_e_dtype_sao_o_contrato_com_o_modelo(self) -> None:
        v = normalizar_mao(mao_sintetica())
        assert v.shape == (N_FEATURES_MAO,)  # 63
        assert v.dtype == np.float32

    def test_shape_errado_falha_ALTO(self) -> None:
        # Preferimos explodir a aceitar silenciosamente. Um array de shape
        # errado que passa despercebido corrompe o dataset inteiro.
        with pytest.raises(ValueError, match="shape"):
            normalizar_mao(np.zeros((10, 3), dtype=np.float32))

    # -- as invariancias que a normalizacao PROMETE ------------------------

    def test_INVARIANTE_A_TRANSLACAO(self) -> None:
        # Mesma mao, deslocada 50 cm. Andar pela sala nao pode mudar o sinal.
        parada = normalizar_mao(mao_sintetica())
        andando = normalizar_mao(mao_sintetica(deslocamento=(0.5, -0.3, 0.2)))
        np.testing.assert_allclose(parada, andando, atol=1e-5)

    def test_INVARIANTE_A_ESCALA(self) -> None:
        # Mao de crianca (60%) e mao grande (140%): mesmo gesto, mesmo vetor.
        # Sem isto, o modelo trataria o tamanho da mao como se fosse o sinal.
        pequena = normalizar_mao(mao_sintetica(escala=0.6))
        grande = normalizar_mao(mao_sintetica(escala=1.4))
        np.testing.assert_allclose(pequena, grande, atol=1e-5)

    def test_INVARIANTE_A_DISTANCIA_DA_CAMERA(self) -> None:
        # Aproximar-se da camera = a mao "cresce" E se desloca. As duas coisas
        # ao mesmo tempo -- que e o caso real.
        longe = normalizar_mao(mao_sintetica(escala=0.7, deslocamento=(0.1, 0.1, 0.8)))
        perto = normalizar_mao(mao_sintetica(escala=1.3, deslocamento=(-0.2, 0.0, 0.2)))
        np.testing.assert_allclose(longe, perto, atol=1e-5)

    # -- a NAO-invariancia que preservamos DE PROPOSITO --------------------

    def test_NAO_e_invariante_a_ROTACAO__e_isso_e_intencional(self) -> None:
        """A orientacao da palma e GRAMATICA em Libras -- ver ADR-012.

        Este teste parece "testar um defeito". Nao e: ele TRAVA uma decisao.
        Se um dia alguem "melhorar" a normalizacao acrescentando invariancia a
        rotacao (parece um upgrade obvio!), este teste quebra e obriga a pessoa
        a ler o ADR antes de apagar um dos cinco parametros da lingua.

        Um teste que protege uma decisao vale mais que um teste que confere
        uma conta.
        """
        world = mao_sintetica()

        # 90 graus em torno do eixo Z (girar a mao no plano da camera).
        theta = np.pi / 2
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        girada = (world @ rot.T).astype(np.float32)

        original = normalizar_mao(world)
        rotacionada = normalizar_mao(girada)

        # DEVEM ser diferentes. Se um dia forem iguais, perdemos a orientacao
        # da palma -- e com ela, a capacidade de distinguir sinais que so
        # diferem por ela.
        assert not np.allclose(original, rotacionada, atol=1e-3)

    # -- robustez ----------------------------------------------------------

    def test_mao_degenerada_devolve_zeros_e_nao_NaN(self) -> None:
        # Se todos os pontos colapsam, a escala e ~0 e a divisao geraria NaN.
        # Um unico NaN se propaga por TODOS os pesos numa passagem de backward
        # e o modelo nunca mais aprende nada. Melhor devolver "nada de util"
        # do que veneno.
        v = normalizar_mao(np.zeros((N_LANDMARKS, 3), dtype=np.float32))
        assert v.shape == (N_FEATURES_MAO,)
        assert np.all(v == 0.0)
        assert not np.any(np.isnan(v))


class TestVetorFeatures:
    def test_shape_fixo_mesmo_sem_mao_nenhuma(self) -> None:
        # Rede neural tem entrada de tamanho FIXO. "Nenhuma mao" nao pode
        # produzir um vetor menor -- produz um vetor de zeros.
        v = vetor_features(ResultadoMaos(maos=()))
        assert v.shape == (N_FEATURES_TOTAL,)  # 128
        assert np.all(v == 0.0)

    def test_a_mao_ocupa_o_bloco_do_SEU_LADO(self) -> None:
        # A ancoragem por LADO (e nao por ordem de deteccao) e o que garante
        # que o mesmo sinal produza sempre o mesmo vetor. O MediaPipe NAO
        # garante a ordem das maos, e ela muda entre frames.
        so_direita = vetor_features(ResultadoMaos(maos=(fazer_mao("Right", mao_sintetica()),)))

        assert np.all(so_direita[0:N_FEATURES_MAO] == 0.0)  # bloco esquerdo vazio
        assert np.any(so_direita[N_FEATURES_MAO : 2 * N_FEATURES_MAO] != 0.0)  # direito cheio

    def test_flags_de_presenca_distinguem_ausencia_de_pose_nula(self) -> None:
        # ESTE e o teste que justifica as flags existirem.
        # Sem elas, "mao ausente" (zeros) e "mao numa pose cujos numeros deram
        # perto de zero" ficariam indistinguiveis para o modelo.
        nenhuma = vetor_features(ResultadoMaos(maos=()))
        assert nenhuma[126] == 0.0 and nenhuma[127] == 0.0

        so_esquerda = vetor_features(ResultadoMaos(maos=(fazer_mao("Left", mao_sintetica()),)))
        assert so_esquerda[126] == 1.0  # esquerda presente
        assert so_esquerda[127] == 0.0  # direita ausente

        duas = vetor_features(
            ResultadoMaos(
                maos=(
                    fazer_mao("Left", mao_sintetica()),
                    fazer_mao("Right", mao_sintetica(deslocamento=(0.3, 0, 0))),
                )
            )
        )
        assert duas[126] == 1.0 and duas[127] == 1.0

    def test_a_ordem_de_deteccao_nao_muda_o_vetor(self) -> None:
        # O mesmo par de maos, entregue em ordens diferentes pelo MediaPipe,
        # TEM que produzir o vetor identico. Se isso falhasse, o mesmo sinal
        # geraria dois vetores distintos e o modelo teria que aprender os dois.
        esq = fazer_mao("Left", mao_sintetica())
        dir_ = fazer_mao("Right", mao_sintetica(deslocamento=(0.3, 0, 0)))

        v1 = vetor_features(ResultadoMaos(maos=(esq, dir_)))
        v2 = vetor_features(ResultadoMaos(maos=(dir_, esq)))

        np.testing.assert_array_equal(v1, v2)


class TestVetorUmaMao:
    """A representacao de 63 dims da datilologia (ADR-014)."""

    def test_sem_mao_devolve_zeros_63(self) -> None:
        v = vetor_uma_mao(ResultadoMaos(maos=()))
        assert v.shape == (N_FEATURES_MAO,)
        assert np.all(v == 0.0)

    def test_uma_mao_e_a_geometria_normalizada(self) -> None:
        world = mao_sintetica()
        v = vetor_uma_mao(ResultadoMaos(maos=(fazer_mao("Right", world),)))
        np.testing.assert_array_equal(v, normalizar_mao(world))

    def test_IGNORA_O_LADO__robusto_ao_erro_de_lateralidade(self) -> None:
        # O ponto central do ADR-014: a MESMA geometria rotulada como "Left" ou
        # "Right" produz features IDENTICAS. E isso que torna a datilologia imune
        # aos ~1-2% de frames em que o MediaPipe erra o lado (visto em C e O nos
        # dados reais).
        world = mao_sintetica()
        como_direita = vetor_uma_mao(ResultadoMaos(maos=(fazer_mao("Right", world),)))
        como_esquerda = vetor_uma_mao(ResultadoMaos(maos=(fazer_mao("Left", world),)))
        np.testing.assert_array_equal(como_direita, como_esquerda)

    def test_com_duas_maos_escolhe_a_de_maior_confianca(self) -> None:
        boa = Mao("Right", "Right", 0.99, np.zeros((N_LANDMARKS, 3), np.float32), mao_sintetica())
        ruim = Mao(
            "Left",
            "Left",
            0.51,
            np.zeros((N_LANDMARKS, 3), np.float32),
            mao_sintetica(deslocamento=(0.5, 0, 0)),
        )
        v = vetor_uma_mao(ResultadoMaos(maos=(ruim, boa)))
        np.testing.assert_array_equal(v, normalizar_mao(boa.world))
