"""Pacote central do Tradutor de Libras.

Este pacote e a UNICA fonte da verdade sobre como um frame vira um vetor de
features. Tanto os scripts de treino (`training/`) quanto o servidor de
inferencia (`backend/`) importam daqui.

A regra e simples e nao se negocia:

    Se uma transformacao e aplicada aos dados durante o TREINO,
    ela precisa vir DESTE pacote -- para que a INFERENCIA aplique
    exatamente a mesma, byte a byte.

Quebrar essa regra produz `training/serving skew`: o modelo recebe em producao
features com distribuicao diferente da que viu no treino, nao levanta excecao,
e simplesmente erra. Ver docs/DECISOES.md (ADR-004).

Subpacotes
----------
vision   : captura de webcam, wrapper do MediaPipe, extracao e normalizacao
           de landmarks.
data     : coleta de amostras, dataset PyTorch, buffer temporal.
models   : definicao das arquiteturas de rede neural.
registry : catalogo declarativo dos sinais suportados.
"""

__version__ = "0.1.0"
