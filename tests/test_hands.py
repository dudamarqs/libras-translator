"""Testes da conversao MediaPipe -> nossos dataclasses.

Nao ha webcam nem modelo .task aqui. `DetectorMaos._converter` e uma funcao
PURA: recebe o objeto do MediaPipe, devolve `ResultadoMaos`. Entao construimos
objetos falsos com a mesma FORMA (duck typing) e testamos a traducao.

Por que vale a pena testar algo tao "simples":

Este e o ponto exato onde os dados entram no nosso sistema. Se `hand_world_landmarks`
for lido na ordem errada, ou se o `handedness` for pego do indice errado, NADA
levanta excecao -- o dataset inteiro sai silenciosamente corrompido, e voce so
descobre depois de treinar um modelo que nao aprende. Fronteiras de dados sao
onde os bugs silenciosos moram, e onde os testes rendem mais.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from libras.vision.hands import (
    CONEXOES,
    N_LANDMARKS,
    NOMES_LANDMARKS,
    DetectorMaos,
    _lado_real,
)


# --- dublês com a mesma forma dos objetos do MediaPipe ----------------------
@dataclass
class FakePonto:
    x: float
    y: float
    z: float


@dataclass
class FakeCategoria:
    category_name: str
    score: float


@dataclass
class FakeResultado:
    hand_landmarks: list[list[FakePonto]]
    hand_world_landmarks: list[list[FakePonto]]
    handedness: list[list[FakeCategoria]]


def _pontos(base: float) -> list[FakePonto]:
    return [FakePonto(base + i, base + i + 0.1, base + i + 0.2) for i in range(N_LANDMARKS)]


class TestConverter:
    def test_sem_maos_produz_resultado_vazio(self) -> None:
        r = DetectorMaos._converter(FakeResultado([], [], []), espelhado=True)
        assert r.vazio
        assert r.maos == ()
        assert r.por_lado("Left") is None

    def test_uma_mao_vira_arrays_numpy_com_shape_correto(self) -> None:
        bruto = FakeResultado(
            hand_landmarks=[_pontos(0.0)],
            hand_world_landmarks=[_pontos(100.0)],
            handedness=[[FakeCategoria("Left", 0.97)]],
        )
        r = DetectorMaos._converter(bruto, espelhado=False)

        assert not r.vazio
        (mao,) = r.maos
        assert mao.lado == "Left"
        assert mao.confianca_lado == pytest.approx(0.97)

        # 21 pontos x 3 eixos -- e o contrato com o resto do pipeline.
        assert mao.landmarks.shape == (N_LANDMARKS, 3)
        assert mao.world.shape == (N_LANDMARKS, 3)
        assert mao.landmarks.dtype == np.float32
        assert mao.world.dtype == np.float32

    def test_nao_confunde_landmarks_normalizados_com_world(self) -> None:
        # O bug que este teste existe para pegar: ler `hand_landmarks` nos DOIS
        # campos. Nada quebraria -- o modelo simplesmente treinaria com features
        # erradas e nunca aprenderia direito. Usamos bases bem distintas (0 e
        # 100) para que a troca seja impossivel de passar despercebida.
        bruto = FakeResultado(
            hand_landmarks=[_pontos(0.0)],
            hand_world_landmarks=[_pontos(100.0)],
            handedness=[[FakeCategoria("Right", 0.9)]],
        )
        (mao,) = DetectorMaos._converter(bruto, espelhado=False).maos

        assert mao.landmarks[0][0] == pytest.approx(0.0)
        assert mao.world[0][0] == pytest.approx(100.0)

    def test_duas_maos_mantem_o_pareamento_entre_lado_e_landmarks(self) -> None:
        # `handedness[i]` tem que casar com `hand_landmarks[i]`. Trocar os
        # indices aqui faria o sistema aprender a mao esquerda com os dados da
        # direita -- e, de novo, sem erro nenhum aparecendo.
        bruto = FakeResultado(
            hand_landmarks=[_pontos(0.0), _pontos(10.0)],
            hand_world_landmarks=[_pontos(100.0), _pontos(200.0)],
            handedness=[[FakeCategoria("Left", 0.9)], [FakeCategoria("Right", 0.8)]],
        )
        r = DetectorMaos._converter(bruto, espelhado=False)

        assert len(r.maos) == 2

        esquerda = r.por_lado("Left")
        direita = r.por_lado("Right")
        assert esquerda is not None and direita is not None

        assert esquerda.landmarks[0][0] == pytest.approx(0.0)
        assert esquerda.world[0][0] == pytest.approx(100.0)
        assert direita.landmarks[0][0] == pytest.approx(10.0)
        assert direita.world[0][0] == pytest.approx(200.0)

    def test_usa_a_categoria_top_1_do_handedness(self) -> None:
        # O MediaPipe devolve uma LISTA de categorias por mao, ordenada por
        # score. Queremos a primeira.
        bruto = FakeResultado(
            hand_landmarks=[_pontos(0.0)],
            hand_world_landmarks=[_pontos(0.0)],
            handedness=[[FakeCategoria("Right", 0.88), FakeCategoria("Left", 0.12)]],
        )
        (mao,) = DetectorMaos._converter(bruto, espelhado=False).maos
        assert mao.lado == "Right"
        assert mao.confianca_lado == pytest.approx(0.88)

    def test_pulso_e_o_landmark_zero(self) -> None:
        bruto = FakeResultado(
            hand_landmarks=[_pontos(0.0)],
            hand_world_landmarks=[_pontos(0.0)],
            handedness=[[FakeCategoria("Left", 1.0)]],
        )
        (mao,) = DetectorMaos._converter(bruto, espelhado=False).maos
        assert np.array_equal(mao.pulso, mao.landmarks[0])


class TestLateralidade:
    """A correcao do espelho.

    Este bug FOI PARA PRODUCAO na primeira versao: o preview mostrava "Left"
    sobre a mao direita do usuario, e o codigo tinha um comentario afirmando que
    isso "nao era um bug". Estes testes existem para que ele nao volte.
    """

    def test_frame_espelhado_inverte_o_lado(self) -> None:
        # A mao direita do usuario, espelhada, tem geometria de mao esquerda ->
        # o MediaPipe responde "Left". A mao REAL e a direita.
        assert _lado_real("Left", espelhado=True) == "Right"
        assert _lado_real("Right", espelhado=True) == "Left"

    def test_frame_nao_espelhado_preserva_o_lado(self) -> None:
        assert _lado_real("Left", espelhado=False) == "Left"
        assert _lado_real("Right", espelhado=False) == "Right"

    def test_categoria_desconhecida_passa_adiante_sem_estourar(self) -> None:
        # Preferimos um rotulo estranho VISIVEL no HUD a um KeyError derrubando
        # o loop de captura inteiro.
        assert _lado_real("Alien", espelhado=True) == "Alien"

    def test_converter_corrige_o_lado_e_preserva_o_bruto(self) -> None:
        bruto = FakeResultado(
            hand_landmarks=[_pontos(0.0)],
            hand_world_landmarks=[_pontos(0.0)],
            handedness=[[FakeCategoria("Left", 0.95)]],
        )
        (mao,) = DetectorMaos._converter(bruto, espelhado=True).maos

        assert mao.lado == "Right"  # a mao real do usuario
        assert mao.lado_bruto == "Left"  # o que o MediaPipe viu, para depuracao

    def test_a_geometria_nao_e_alterada_pela_correcao(self) -> None:
        # Trocamos o ROTULO, nunca a GEOMETRIA. Os landmarks continuam vindo da
        # imagem espelhada -- e e isso que preserva a consistencia entre a
        # coleta e a inferencia.
        bruto = FakeResultado(
            hand_landmarks=[_pontos(0.0)],
            hand_world_landmarks=[_pontos(100.0)],
            handedness=[[FakeCategoria("Left", 0.95)]],
        )
        (espelhada,) = DetectorMaos._converter(bruto, espelhado=True).maos
        (crua,) = DetectorMaos._converter(bruto, espelhado=False).maos

        assert espelhada.lado != crua.lado
        assert np.array_equal(espelhada.landmarks, crua.landmarks)
        assert np.array_equal(espelhada.world, crua.world)


class TestConstantes:
    def test_ha_exatamente_21_nomes_de_landmark(self) -> None:
        assert len(NOMES_LANDMARKS) == N_LANDMARKS
        assert NOMES_LANDMARKS[0] == "PULSO"
        # As 5 pontas de dedo -- os pontos que mais distinguem um sinal do outro.
        for i in (4, 8, 12, 16, 20):
            assert NOMES_LANDMARKS[i].endswith("PONTA")

    def test_conexoes_referenciam_apenas_landmarks_validos(self) -> None:
        # Um indice fora do intervalo aqui so estouraria na hora de DESENHAR --
        # ou seja, em producao, na frente do usuario.
        for inicio, fim in CONEXOES:
            assert 0 <= inicio < N_LANDMARKS
            assert 0 <= fim < N_LANDMARKS
            assert inicio != fim
