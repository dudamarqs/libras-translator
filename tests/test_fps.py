"""Testes do medidor de FPS e do cronometro.

Repare no que estamos testando: logica PURA, sem webcam nenhuma. Isso e
consequencia direta de `MedidorFPS` e `Cronometro` viverem separados da
`Camera`. Se o medidor de FPS estivesse embutido no loop de captura, testa-lo
exigiria uma webcam -- e um teste que depende de hardware nao roda no CI, falha
aleatoriamente e acaba sendo desligado por alguem cansado.

**Codigo testavel nao e um acidente feliz; e o que sobra quando as
responsabilidades estao separadas.**
"""

from __future__ import annotations

import time

import pytest

from libras.vision.fps import Cronometro, MedidorFPS


class TestMedidorFPS:
    def test_sem_frames_suficientes_retorna_zero(self) -> None:
        # 1 marca delimita 0 intervalos -- nao da para calcular taxa nenhuma.
        # Retornar 0.0 (e nao dividir por zero) e o contrato.
        medidor = MedidorFPS()
        assert medidor.fps == 0.0

        medidor.marcar()
        assert medidor.fps == 0.0

    def test_calcula_fps_a_partir_de_marcas_espacadas(self, monkeypatch) -> None:
        # Nao usamos time.sleep(): um teste que dorme e um teste lento e
        # instavel. Substituimos o relogio por uma sequencia controlada --
        # a mesma tecnica que usamos com um relogio falso em qualquer sistema
        # que dependa de tempo.
        relogio = iter([0.0, 0.1, 0.2, 0.3, 0.4])  # 100ms entre frames = 10 FPS
        monkeypatch.setattr(time, "perf_counter", lambda: next(relogio))

        medidor = MedidorFPS()
        for _ in range(5):
            medidor.marcar()

        # 5 marcas, 4 intervalos, 0.4s no total -> 4 / 0.4 = 10 FPS
        assert medidor.fps == pytest.approx(10.0)

    def test_janela_descarta_marcas_antigas(self, monkeypatch) -> None:
        # A janela e o que impede o vazamento de memoria: sem `maxlen`, o deque
        # cresceria para sempre num loop que roda por horas.
        relogio = iter([float(i) for i in range(10)])
        monkeypatch.setattr(time, "perf_counter", lambda: next(relogio))

        medidor = MedidorFPS(janela=3)
        for _ in range(10):
            medidor.marcar()

        # Guardou apenas as 3 ultimas marcas (7.0, 8.0, 9.0):
        # 2 intervalos em 2 segundos -> 1 FPS.
        assert medidor.fps == pytest.approx(1.0)

    def test_intervalo_nulo_nao_divide_por_zero(self, monkeypatch) -> None:
        # Relogios de baixa resolucao podem devolver o mesmo valor duas vezes.
        # Uma divisao por zero aqui derrubaria o loop de captura inteiro por
        # causa de um NUMERO DE DIAGNOSTICO. Metrica nunca deve matar o sistema.
        monkeypatch.setattr(time, "perf_counter", lambda: 42.0)

        medidor = MedidorFPS()
        medidor.marcar()
        medidor.marcar()

        assert medidor.fps == 0.0


class TestCronometro:
    def test_mede_etapas_separadamente(self, monkeypatch) -> None:
        # captura: 0.000 -> 0.002  = 2ms
        # modelo : 0.002 -> 0.027  = 25ms
        relogio = iter([0.0, 0.002, 0.002, 0.027])
        monkeypatch.setattr(time, "perf_counter", lambda: next(relogio))

        crono = Cronometro()
        with crono.medir("captura"):
            pass
        with crono.medir("modelo"):
            pass

        resumo = crono.resumo()
        assert resumo["captura"] == pytest.approx(2.0)
        assert resumo["modelo"] == pytest.approx(25.0)

    def test_registra_tempo_mesmo_se_a_etapa_falhar(self, monkeypatch) -> None:
        # O `finally` no context manager existe por isto: um erro intermitente
        # sumiria das metricas justamente nos frames que mais interessam.
        relogio = iter([0.0, 0.005])
        monkeypatch.setattr(time, "perf_counter", lambda: next(relogio))

        crono = Cronometro()
        with pytest.raises(ValueError), crono.medir("etapa_que_falha"):
            raise ValueError("boom")

        assert crono.resumo()["etapa_que_falha"] == pytest.approx(5.0)

    def test_resumo_vazio_sem_medicoes(self) -> None:
        assert Cronometro().resumo() == {}
