"""Transformacao de landmarks em FEATURES -- o vetor que o modelo recebe.

Este e o modulo mais critico do projeto inteiro, e o motivo e o ADR-004:

    O `training/collect.py` e o `backend/` importam ESTAS funcoes.
    Se a normalizacao do treino divergir da normalizacao da inferencia, o
    modelo recebe em producao features com distribuicao diferente da que viu no
    treino. Ele NAO levanta excecao. Ele so passa a errar.

Por isso tudo aqui e funcao pura, testada, e vive num lugar so.

--------------------------------------------------------------------------
O QUE NORMALIZAMOS, E O QUE DEIXAMOS EM PAZ
--------------------------------------------------------------------------

Partimos de `Mao.world`: (21, 3) em METROS, origem no centro da mao. Ele ja
resolve a POSICAO no quadro -- andar para a esquerda nao muda os numeros.
Sobram dois problemas:

1. ESCALA. Os valores estao em metros REAIS. Uma crianca tem a mao ~30% menor:
   mesmo sinal, todos os numeros 30% menores, e o modelo veria dois gestos
   diferentes. => Dividimos por uma distancia de referencia da propria mao.
   A mao vira ADIMENSIONAL: importa a PROPORCAO entre os pontos, nao o tamanho.

2. ROTACAO. Inclinar a mao muda todos os numeros. A saida obvia seria girar a
   mao para uma orientacao canonica.

   NAO FAZEMOS ISSO. E a decisao mais importante deste arquivo, e ela nao e
   tecnica -- e LINGUISTICA.

   Um sinal em Libras tem cinco parametros: configuracao de mao, ponto de
   articulacao, movimento, ORIENTACAO DA PALMA e expressao facial. A orientacao
   da palma e GRAMATICA: palma para dentro e palma para fora podem ser sinais
   DIFERENTES. Normalizar a rotacao apagaria um dos cinco parametros da lingua,
   e o modelo ficaria cego para uma distincao que os sinalizantes usam.

   Ver ADR-012.
"""

from __future__ import annotations

import numpy as np

# landmarks (leve, sem MediaPipe): treino e pre-processamento importam features
# e nao podem arrastar o MediaPipe junto. Ver o cabecalho de landmarks.py.
from libras.vision.landmarks import N_LANDMARKS, Mao, ResultadoMaos

PULSO = 0
MEDIO_MCP = 9  # a base do dedo medio -- o "centro" estavel da palma

# 21 pontos x 3 eixos.
N_FEATURES_MAO = N_LANDMARKS * 3  # 63

# Duas maos + 2 flags de presenca. Ver `vetor_features`.
N_FEATURES_TOTAL = 2 * N_FEATURES_MAO + 2  # 128

# Abaixo disto, a "mao" e degenerada (landmarks colapsados num ponto) e dividir
# por ela produziria numeros absurdos ou NaN. Em metros.
_ESCALA_MINIMA = 1e-6


def escala_da_mao(world: np.ndarray) -> float:
    """Distancia PULSO -> base do dedo medio. E a nossa "regua" da mao.

    Por que ESTES dois pontos, e nao outros:

    Precisamos de uma distancia que dependa SO do tamanho da mao, e nunca do
    gesto. Pulso -> base do medio atravessa a PALMA, que e rigida: ela mede o
    mesmo quer voce feche o punho, abra a mao ou faca um "joia".

    Contra-exemplo do que NAO funciona: pulso -> ponta do medio. Essa distancia
    muda de ~18 cm (mao aberta) para ~8 cm (punho fechado) -- ou seja, ela
    depende do GESTO. Usa-la como regua faria o punho fechado parecer uma mao
    aberta gigante. A regua tem que ser independente daquilo que ela mede.
    """
    return float(np.linalg.norm(world[MEDIO_MCP] - world[PULSO]))


def normalizar_mao(world: np.ndarray) -> np.ndarray:
    """(21, 3) em metros -> (63,) adimensional, pronto para o modelo.

    Duas operacoes, nesta ordem:

    1. Recentrar no PULSO. O `world` do MediaPipe ja vem centrado no "centro
       geometrico da mao" -- mas esse centro se MOVE conforme os dedos dobram
       (e uma media dos 21 pontos). Um referencial que se mexe nao e
       referencial. O pulso e uma ancora ANATOMICA: sempre o mesmo lugar do
       corpo, independentemente do gesto.

    2. Dividir pela escala da mao. Torna o vetor invariante ao TAMANHO da mao
       (adulto vs crianca, mao perto vs longe da camera).

    A rotacao e DELIBERADAMENTE preservada -- ver o cabecalho do modulo.
    """
    if world.shape != (N_LANDMARKS, 3):
        raise ValueError(f"esperado shape ({N_LANDMARKS}, 3), recebido {world.shape}")

    centrado = world - world[PULSO]

    escala = escala_da_mao(world)
    if escala < _ESCALA_MINIMA:
        # Mao degenerada (todos os pontos no mesmo lugar). Dividir aqui geraria
        # inf/NaN, que envenenariam o gradiente do treino inteiro -- um NaN se
        # propaga por TODOS os pesos numa unica passagem de backward, e o modelo
        # nunca mais aprende nada. Melhor devolver zeros: o modelo ve "nada de
        # util" em vez de "veneno".
        return np.zeros(N_FEATURES_MAO, dtype=np.float32)

    return (centrado / escala).astype(np.float32).ravel()


def vetor_features(resultado: ResultadoMaos) -> np.ndarray:
    """ResultadoMaos -> (128,) float32. A ENTRADA DO MODELO.

    Layout, fixo e explicito:

        [  0.. 62]  mao esquerda  (63 = 21 pontos x 3 eixos, normalizados)
        [ 63..125]  mao direita   (63)
        [126]       flag: mao esquerda presente? (0.0 ou 1.0)
        [127]       flag: mao direita presente?  (0.0 ou 1.0)

    TRES decisoes embutidas aqui, todas com armadilha:

    1. POSICAO FIXA POR LADO. A mao esquerda SEMPRE ocupa as posicoes 0..62.
       Nao e "a primeira mao detectada". O MediaPipe nao garante ordem, e a
       ordem MUDA entre frames. Se voce usasse a ordem de deteccao, o mesmo
       sinal produziria vetores diferentes em frames diferentes -- e o modelo
       teria que aprender as duas versoes. Ancorar por LADO elimina isso.
       (Isto so funciona porque corrigimos a lateralidade -- ADR-011.)

    2. MAO AUSENTE = ZEROS. Muitos sinais usam so uma mao. O vetor tem que ter
       SEMPRE o mesmo tamanho (uma rede neural tem entrada de tamanho fixo),
       entao a mao que falta vira zeros.

    3. AS FLAGS DE PRESENCA EXISTEM POR CAUSA DA DECISAO 2 -- e sao a parte que
       quase todo mundo esquece. Sem elas, o modelo nao consegue distinguir
       "esta mao nao esta no quadro" de "esta mao esta numa pose cujos numeros
       normalizados deram perto de zero". Sao duas situacoes MUITO diferentes e
       o vetor as representaria de forma parecida. As flags dizem ao modelo,
       sem ambiguidade: este bloco e informacao, ou este bloco e vazio.
    """
    features = np.zeros(N_FEATURES_TOTAL, dtype=np.float32)

    esquerda = resultado.por_lado("Left")
    direita = resultado.por_lado("Right")

    if esquerda is not None:
        features[0:N_FEATURES_MAO] = normalizar_mao(esquerda.world)
        features[2 * N_FEATURES_MAO] = 1.0

    if direita is not None:
        features[N_FEATURES_MAO : 2 * N_FEATURES_MAO] = normalizar_mao(direita.world)
        features[2 * N_FEATURES_MAO + 1] = 1.0

    return features


def vetor_uma_mao(resultado: ResultadoMaos) -> np.ndarray:
    """ResultadoMaos -> (63,) float32. A entrada do modelo para sinais de UMA mao.

    Esta e a representacao da DATILOLOGIA (e de qualquer sinal de uma mao).

    POR QUE NAO USAMOS O VETOR DE 128 (`vetor_features`) AQUI -- e a licao que os
    DADOS REAIS nos ensinaram:

    O vetor de 128 ancora cada mao no bloco do seu LADO (esquerda 0..62, direita
    63..125). Isso e essencial para sinais de DUAS maos, onde e preciso saber
    qual e qual. Mas o classificador de lateralidade do MediaPipe erra ~1-2% dos
    frames -- justamente nas maos "redondas" e quase simetricas (o C e o O foram
    os que mais confundiram na coleta real). Num vetor de 128, esses frames
    trocados cairiam no bloco errado, e o modelo veria a MESMA letra em duas
    regioes diferentes do vetor. Em producao, um flip do MediaPipe mandaria a
    predicao para o buraco.

    Para um sinal de uma mao, o LADO e ruido: o que distingue as letras e a
    GEOMETRIA da mao, que `normalizar_mao` extrai sem olhar o lado nenhuma vez.
    Entao usamos os 63 numeros da mao detectada e ignoramos qual lado o
    MediaPipe achou que era. Robusto ao erro de lateralidade por construcao.

    Ver ADR-014.

    (Limitacao conhecida: um modelo treinado com a mao direita ve a mao esquerda
    ESPELHADA e nao a reconhece. Sinalize com a mesma mao que treinou. Na Etapa 8
    resolvemos isso de graca com data augmentation: espelhar o eixo x sintetiza
    a outra mao.)
    """
    if resultado.vazio:
        return np.zeros(N_FEATURES_MAO, dtype=np.float32)
    # So ha uma mao (datilologia usa max_maos=1). Se por acaso vierem duas,
    # a de maior confianca de lado e a aposta mais segura.
    mao = max(resultado.maos, key=lambda m: m.confianca_lado)
    return normalizar_mao(mao.world)


def features_de_mao(mao: Mao) -> np.ndarray:
    """Atalho para inspecionar uma mao isolada (analise, notebooks, testes)."""
    return normalizar_mao(mao.world)
