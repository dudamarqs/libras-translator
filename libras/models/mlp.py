"""Rede neural (MLP) para classificar a datilologia -- Etapa 9.

Por que uma MLP entra AGORA, e nao antes: o baseline LogReg (ADR-002) chegou ao
teto nos clusters de letras parecidas (R/U/V, T/F, M/N) -- fronteiras que uma
reta nao separa. A MLP existe pelos DADOS (a matriz de confusao), nao pelo
roteiro. Se ela nao bater o baseline, o baseline vence -- e isso tambem e um
resultado honesto.

Este arquivo tem DUAS partes:
  1. RedeMLP        -- a arquitetura (o que a rede E).
  2. treinar_rede   -- o loop de treino EXPLICITO (forward/loss/backward/step).
                       Escrito a mao de proposito: e onde se ve o backprop
                       acontecer, que e o motivo de termos escolhido PyTorch.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# CPU e suficiente: a rede tem ~17 mil pesos e o dataset cabe na memoria.
# (Confirmamos no doctor.py que nao ha GPU CUDA -- e nao faz falta aqui.)
_DEVICE = torch.device("cpu")


class RedeMLP(nn.Module):
    """Multi-Layer Perceptron: 63 -> ocultas -> n_classes.

    nn.Module e a classe-base de todo modelo em PyTorch. Duas coisas importam:
      - __init__ DECLARA as camadas (os pesos que serao aprendidos).
      - forward  DEFINE como um dado atravessa essas camadas.
    O PyTorch cuida do backward sozinho a partir do forward (autograd).
    """

    def __init__(
        self,
        n_entrada: int = 63,
        n_classes: int = 20,
        ocultas: tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        camadas: list[nn.Module] = []
        anterior = n_entrada
        for h in ocultas:
            camadas.append(nn.Linear(anterior, h))
            # ReLU = max(0, x). E ela que quebra a linearidade: sem uma funcao
            # de ativacao nao-linear entre as camadas Linear, tres Lineares
            # colapsam numa unica transformacao linear -- seria uma LogReg cara.
            camadas.append(nn.ReLU())
            # Dropout: no treino, zera aleatoriamente 30% dos neuronios desta
            # camada a cada passada. Forca a rede a NAO depender de um neuronio
            # so -> regularizacao -> menos overfitting. No modo eval ele some
            # (nn.Module.eval desliga o dropout automaticamente).
            camadas.append(nn.Dropout(dropout))
            anterior = h

        # Camada de saida: um logit por classe. SEM softmax aqui -- a
        # CrossEntropyLoss aplica o softmax internamente, de forma
        # numericamente estavel. Por um softmax a mais aqui seria um bug sutil
        # (softmax duas vezes) que so apareceria como um modelo "preguicoso".
        camadas.append(nn.Linear(anterior, n_classes))

        self.rede = nn.Sequential(*camadas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rede(x)


def treinar_rede(
    X_treino: np.ndarray,
    y_treino: np.ndarray,
    *,
    n_classes: int,
    ocultas: tuple[int, ...] = (128, 64),
    dropout: float = 0.3,
    epocas: int = 200,
    lote: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    semente: int = 0,
    monitorar: Callable[[int, float], None] | None = None,
) -> RedeMLP:
    """O loop de treino explicito. Recebe features JA ESCALADAS e rotulos em
    indices (0..n_classes-1), devolve a rede treinada.

    weight_decay: penaliza pesos grandes (regularizacao L2). Junto com o
    dropout, e a segunda defesa contra o overfitting -- pesos menores = fronteira
    mais suave = generaliza melhor.
    """
    # Reprodutibilidade: rede neural e ESTOCASTICA (init aleatorio dos pesos,
    # ordem dos lotes, dropout). Sem fixar a semente, dois treinos dao numeros
    # diferentes e voce nunca sabe se uma mudanca ajudou ou foi sorte.
    torch.manual_seed(semente)

    modelo = RedeMLP(
        n_entrada=X_treino.shape[1],
        n_classes=n_classes,
        ocultas=ocultas,
        dropout=dropout,
    ).to(_DEVICE)

    Xt = torch.tensor(X_treino, dtype=torch.float32, device=_DEVICE)
    yt = torch.tensor(y_treino, dtype=torch.long, device=_DEVICE)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=lote, shuffle=True)

    # Adam: gradiente descendente com passo adaptativo por parametro.
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    # CrossEntropyLoss = softmax + log-verossimilhanca negativa, num passo estavel.
    criterio = nn.CrossEntropyLoss()

    modelo.train()  # liga o dropout
    for epoca in range(epocas):
        perda_acumulada = 0.0
        for xb, yb in loader:
            # ---- o coracao de todo treino em PyTorch, quatro linhas ----
            otimizador.zero_grad()  # zera gradientes da iteracao anterior
            #                         (o PyTorch os ACUMULA por padrao; esquecer
            #                         isto e o bug classico numero 1)
            logits = modelo(xb)  # 1. FORWARD
            perda = criterio(logits, yb)  # 2. LOSS
            perda.backward()  # 3. BACKWARD (autograd -> gradientes)
            otimizador.step()  # 4. STEP (ajusta os pesos)
            # ------------------------------------------------------------
            perda_acumulada += perda.item() * len(xb)

        if monitorar is not None:
            monitorar(epoca, perda_acumulada / len(Xt))

    return modelo


def prever_indices(modelo: RedeMLP, X: np.ndarray) -> np.ndarray:
    """Prediz os indices de classe para X (features ja escaladas)."""
    modelo.eval()  # desliga o dropout -- em producao usamos a rede inteira
    with torch.no_grad():  # sem gradientes na inferencia: mais rapido, menos memoria
        logits = modelo(torch.tensor(X, dtype=torch.float32, device=_DEVICE))
        return logits.argmax(dim=1).cpu().numpy()
