"""Testes do GravadorGif.

Nenhum `sleep` aqui: o relogio e injetado. Um teste que dorme para "esperar o
proximo quadro" e lento e mentiroso -- passa na sua maquina e falha na CI
carregada. Controlando o tempo, o teste verifica a REGRA (um quadro a cada
1/fps segundos) em vez de torcer para o agendador do sistema colaborar.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from libras.vision.gravador import GravadorGif


class RelogioFalso:
    """Relogio que so anda quando o teste manda."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def avancar(self, segundos: float) -> None:
        self.t += segundos


def frame_qualquer(cor: int = 30, largura: int = 640, altura: int = 480) -> np.ndarray:
    return np.full((altura, largura, 3), cor, dtype=np.uint8)


def test_ignora_frames_enquanto_parado(tmp_path):
    grav = GravadorGif(tmp_path / "x.gif", relogio=RelogioFalso())

    for _ in range(10):
        assert grav.capturar(frame_qualquer()) is False

    assert grav.n_quadros == 0
    assert grav.salvar() is None  # sem quadros nao existe GIF -- nao um arquivo vazio


def test_amostra_pelo_relogio_e_nao_por_contagem_de_frames(tmp_path):
    """60 frames em 1 segundo, a 10 q/s, tem que virar ~10 quadros."""
    relogio = RelogioFalso()
    grav = GravadorGif(tmp_path / "x.gif", fps=10, relogio=relogio)
    grav.alternar()

    guardados = 0
    for _ in range(60):  # um segundo de camera a 60 FPS
        guardados += grav.capturar(frame_qualquer())
        relogio.avancar(1 / 60)

    assert guardados == 10
    assert grav.segundos == pytest.approx(1.0)


def test_loop_lento_nao_vira_camera_lenta(tmp_path):
    """Se o loop entrega 4 frames por segundo, o GIF guarda os 4 -- nao espera 10.

    E o caso real do MediaPipe engasgando: o GIF fica mais curto, mas o que
    aparece nele continua no ritmo do relogio.
    """
    relogio = RelogioFalso()
    grav = GravadorGif(tmp_path / "x.gif", fps=10, relogio=relogio)
    grav.alternar()

    for _ in range(4):
        assert grav.capturar(frame_qualquer()) is True
        relogio.avancar(0.25)

    assert grav.n_quadros == 4


def test_para_sozinho_no_limite_de_seguranca(tmp_path):
    relogio = RelogioFalso()
    grav = GravadorGif(tmp_path / "x.gif", fps=10, segundos_max=2.0, relogio=relogio)
    grav.alternar()

    for _ in range(100):
        grav.capturar(frame_qualquer())
        relogio.avancar(0.1)

    assert grav.ativo is False
    assert grav.n_quadros == 20  # 2 s a 10 q/s, e nem um quadro a mais


def test_religar_nao_perde_o_primeiro_quadro(tmp_path):
    relogio = RelogioFalso()
    grav = GravadorGif(tmp_path / "x.gif", fps=10, relogio=relogio)

    grav.alternar()
    assert grav.capturar(frame_qualquer()) is True
    grav.alternar()  # parou

    relogio.avancar(30.0)  # muito tempo depois
    grav.alternar()  # religou
    assert grav.capturar(frame_qualquer()) is True


def test_gif_gerado_e_animado_e_no_ritmo_certo(tmp_path):
    relogio = RelogioFalso()
    destino = tmp_path / "sub" / "demo.gif"  # a pasta ainda nao existe
    grav = GravadorGif(destino, fps=10, largura=320, relogio=relogio)
    grav.alternar()

    for i in range(5):
        grav.capturar(frame_qualquer(cor=30 + i * 40))
        relogio.avancar(0.1)

    assert grav.salvar() == destino
    with Image.open(destino) as gif:
        assert gif.n_frames == 5
        assert gif.info["duration"] == 100  # ms por quadro = 1000/fps
        assert gif.info["loop"] == 0  # repete para sempre
        assert gif.size == (320, 240)  # reduzido mantendo a proporcao 4:3


def test_frame_menor_que_o_alvo_nao_e_ampliado(tmp_path):
    """Ampliar so gastaria bytes: nao ha detalhe novo para mostrar."""
    relogio = RelogioFalso()
    destino = tmp_path / "p.gif"
    grav = GravadorGif(destino, fps=10, largura=560, relogio=relogio)
    grav.alternar()
    grav.capturar(frame_qualquer(largura=320, altura=240))
    grav.salvar()

    with Image.open(destino) as gif:
        assert gif.size == (320, 240)


def test_estabilizador_congela_ruido_mas_deixa_o_movimento_passar(tmp_path):
    """O criterio e por pixel: parede tremendo congela, mao movendo nao.

    Este e o teste que protege o tamanho do arquivo. Se alguem subir a
    tolerancia sem querer, a mao comeca a ser congelada junto e o GIF fica
    borrado; se descer, o ruido volta e o arquivo incha.
    """
    rng = np.random.default_rng(7)
    grav = GravadorGif(tmp_path / "x.gif", relogio=RelogioFalso(), tolerancia_ruido=10)

    parede = np.full((48, 64, 3), 120, dtype=np.uint8)
    primeiro = grav._estabilizar(parede.copy())

    # so ruido: +-3 niveis, abaixo da tolerancia
    ruidoso = np.clip(parede.astype(np.int16) + rng.integers(-3, 4, parede.shape), 0, 255).astype(
        np.uint8
    )
    # a "mao": um bloco que mudou MUITO
    ruidoso[10:20, 10:20] = 240

    saida = grav._estabilizar(ruidoso)

    fundo = np.ones(saida.shape[:2], bool)
    fundo[10:20, 10:20] = False
    assert np.array_equal(saida[fundo], primeiro[fundo])  # ruido congelado
    assert np.all(saida[10:20, 10:20] == 240)  # movimento preservado


def test_tolerancia_zero_desliga_o_estabilizador(tmp_path):
    grav = GravadorGif(tmp_path / "x.gif", relogio=RelogioFalso(), tolerancia_ruido=0)
    grav._estabilizar(np.full((8, 8, 3), 100, dtype=np.uint8))
    quase_igual = np.full((8, 8, 3), 101, dtype=np.uint8)
    assert np.all(grav._estabilizar(quase_igual) == 101)


def test_primeiro_quadro_ruim_nao_sequestra_a_paleta(tmp_path):
    """Regressao: a paleta sai da AMOSTRA, nao do primeiro quadro.

    O bug real: a camera abre escura (exposicao ajustando) e a mao ainda nem
    entrou em cena. Tirando a paleta dali, o clipe inteiro era projetado nas
    cores do escuro -- quadros visivelmente diferentes viravam o MESMO quadro, e
    o Pillow os fundia. O GIF de 8 quadros saia com 1.
    """
    relogio = RelogioFalso()
    destino = tmp_path / "escuro.gif"
    grav = GravadorGif(destino, fps=10, largura=64, relogio=relogio)
    grav.alternar()

    grav.capturar(frame_qualquer(cor=0, largura=64, altura=48))  # o quadro ruim
    relogio.avancar(0.1)
    for i in range(1, 8):
        grav.capturar(frame_qualquer(cor=i * 30, largura=64, altura=48))
        relogio.avancar(0.1)

    grav.salvar()
    with Image.open(destino) as gif:
        assert gif.n_frames == 8


def test_cores_do_hud_sobrevivem_a_paleta(tmp_path):
    """Regressao: o HUD perde a votacao da paleta se nao for reservado.

    O quantizador escolhe cores por frequencia. Numa cena de rosto e parede, o
    HUD sao poucos milhares de pixels contra centenas de milhares -- e as cores
    dele eram absorvidas pelo tom de pele vizinho. Medido na primeira gravacao
    real: o amarelo (250,200,60) virou (214,212,193), um cinza. No demo a cor E
    a informacao (verde = aceito, vermelho = desconhecido), entao isso apagava
    justamente o que o GIF existe para mostrar.
    """
    rng = np.random.default_rng(3)
    # cena "de pele e parede": muitos tons quentes proximos, nenhum saturado
    cena = rng.integers(150, 210, (240, 320, 3), dtype=np.int16)
    cena[:, :, 2] -= 40  # puxa para o quente
    cena = cena.astype(np.uint8)

    verde_bgr, amarelo_bgr, vermelho_bgr = (80, 220, 100), (60, 200, 250), (70, 70, 240)
    cena[10:40, 10:120] = verde_bgr
    cena[50:80, 10:120] = amarelo_bgr
    cena[90:120, 10:120] = vermelho_bgr

    destino = tmp_path / "hud.gif"
    grav = GravadorGif(destino, fps=10, largura=320, relogio=RelogioFalso())
    grav.alternar()
    grav.capturar(cena)
    grav.salvar()

    saida = np.asarray(Image.open(destino).convert("RGB")).astype(int)
    for nome, bgr in [("verde", verde_bgr), ("amarelo", amarelo_bgr), ("vermelho", vermelho_bgr)]:
        alvo = np.array(list(reversed(bgr)))
        exatos = int((np.abs(saida - alvo).sum(axis=2) == 0).sum())
        assert exatos > 500, f"{nome} nao sobreviveu a paleta ({exatos} pixels exatos)"


def test_fps_invalido_falha_na_criacao(tmp_path):
    with pytest.raises(ValueError, match="fps"):
        GravadorGif(tmp_path / "x.gif", fps=0)
