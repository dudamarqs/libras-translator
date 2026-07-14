"""Medicao de FPS e de tempo gasto por etapa do pipeline.

Por que isso e um modulo, e nao tres linhas soltas no loop:

Voce vai otimizar este sistema (Etapa 16). Otimizar sem medir e chute. E, mais
importante, "o loop roda a 30 FPS" nao te diz NADA sobre onde o tempo e gasto:
a captura pode custar 2ms e o MediaPipe 28ms -- ou o contrario. Quando a Etapa 4
adicionar o MediaPipe e o FPS cair, voce vai querer saber exatamente quem comeu
o orcamento, e nao adivinhar.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager


class MedidorFPS:
    """FPS suavizado por media movel.

    Por que uma media movel e nao o FPS instantaneo (1 / delta do ultimo frame):

    O FPS instantaneo oscila violentamente. Um unico frame que demorou 50ms --
    porque o Windows resolveu indexar um arquivo naquele milissegundo -- faz o
    numero despencar de 30 para 20 e voltar. Na tela vira um borrao ilegivel, e
    voce nao consegue dizer se seu codigo ficou mais lento ou se foi ruido.

    A media sobre uma janela de N frames absorve esse ruido e mostra a
    tendencia, que e o que voce realmente quer saber.

    Guardamos os TIMESTAMPS (nao os deltas) porque o FPS e entao simplesmente
    "quantos frames couberam no intervalo entre o mais antigo e o mais novo" --
    uma divisao, sem acumular erro de ponto flutuante.
    """

    def __init__(self, janela: int = 30) -> None:
        # maxlen faz o deque descartar o mais antigo automaticamente. Sem isso,
        # a lista cresceria para sempre -- um vazamento de memoria lento, do tipo
        # que so aparece depois de horas rodando.
        self._marcas: deque[float] = deque(maxlen=janela)

    def marcar(self) -> None:
        """Registra que um frame acabou de ser processado."""
        # perf_counter, nao time.time(). time.time() e o relogio de PAREDE:
        # ele pode andar para tras (ajuste de NTP, horario de verao) e produzir
        # um delta negativo. perf_counter e monotonico -- so anda para frente --
        # e tem resolucao de nanossegundos. Para medir DURACAO, e sempre ele.
        self._marcas.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._marcas) < 2:
            return 0.0
        intervalo = self._marcas[-1] - self._marcas[0]
        if intervalo <= 0:
            return 0.0
        # N marcas delimitam N-1 intervalos. Usar N aqui superestimaria o FPS.
        return (len(self._marcas) - 1) / intervalo


class Cronometro:
    """Mede quanto tempo cada etapa do pipeline consome, em milissegundos.

        crono = Cronometro()
        with crono.medir("captura"):
            frame = cam.ler()
        with crono.medir("mediapipe"):
            landmarks = detector.detectar(frame)

        crono.resumo()  # {'captura': 1.8, 'mediapipe': 24.3}

    Cada etapa tem sua propria media movel, entao o resumo mostra o orcamento
    de tempo do frame repartido por responsavel.
    """

    def __init__(self, janela: int = 30) -> None:
        self._janela = janela
        self._amostras: dict[str, deque[float]] = {}

    @contextmanager
    def medir(self, etapa: str) -> Iterator[None]:
        inicio = time.perf_counter()
        try:
            yield
        finally:
            # `finally`: se a etapa levantar excecao, ainda assim registramos o
            # tempo. Sem isso, um erro intermitente sumiria das metricas
            # justamente nos frames que mais interessam.
            decorrido_ms = (time.perf_counter() - inicio) * 1000.0
            self._amostras.setdefault(etapa, deque(maxlen=self._janela)).append(decorrido_ms)

    def resumo(self) -> dict[str, float]:
        """Media, em ms, de cada etapa medida."""
        return {
            etapa: sum(amostras) / len(amostras)
            for etapa, amostras in self._amostras.items()
            if amostras
        }
