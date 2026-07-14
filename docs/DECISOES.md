# Registro de Decisões de Arquitetura (ADR)

Cada entrada responde: **qual era o problema**, **o que decidimos**, **o que
descartamos e por quê**, e **o que essa decisão nos custa**.

Isso existe porque, daqui a seis meses, ninguém (incluindo você) lembra por que
o `numpy` está travado numa versão específica — e alguém vai "arrumar" isso e
quebrar o projeto.

---

## ADR-001 — Classificar landmarks, não pixels

**Data:** 2026-07-14 · **Status:** aceito

**Problema.** Como transformar um frame de webcam num sinal de Libras
reconhecido?

**Alternativas.**

1. **CNN sobre a imagem crua.** Entrada de 224×224×3 = 150.528 valores. A rede
   aprende sozinha a extrair as features. Exige milhares de exemplos por sinal,
   aprende junto o fundo/iluminação/aparência do sinalizante, e é pesada para
   rodar a 30 FPS em CPU.
2. **MediaPipe → landmarks → classificador leve.** O MediaPipe (rede
   pré-treinada pelo Google) devolve as coordenadas 3D das articulações.
   O frame vira um vetor de ~126 números que descreve apenas a **geometria**
   do gesto.

**Decisão: alternativa 2.**

**Racional.** O MediaPipe atua como **extrator de features pré-treinado** —
isto é, *transfer learning*. A representação que ele entrega já é invariante a
fundo, iluminação e cor de pele, porque foi treinada com milhões de imagens.
Consequências práticas: ~100–200 amostras por sinal em vez de milhares;
treino em segundos, sem GPU; inferência trivialmente em tempo real.

**Custo.** Ficamos cegos ao que o MediaPipe não vê. Se ele falha em detectar a
mão (oclusão, movimento muito rápido, mão fora do quadro), nosso classificador
não recebe nada — não há degradação suave. A qualidade do nosso sistema tem
como teto a qualidade do MediaPipe.

---

## ADR-002 — PyTorch como framework de deep learning

**Data:** 2026-07-14 · **Status:** aceito

**Problema.** PyTorch ou TensorFlow/Keras?

**Decisão: PyTorch**, com `scikit-learn` como baseline obrigatório antes de
qualquer rede neural.

**Racional.**

- O loop de treino é **explícito** (`forward → loss → backward → step`). Como o
  objetivo do projeto é *entender* redes neurais, ver o backprop acontecer é
  uma vantagem pedagógica; o `model.fit()` do Keras esconde exatamente aquilo
  que queremos aprender.
- Execução eager: dá para pôr `breakpoint()` no meio do `forward` e inspecionar
  tensores.
- Predominante em pesquisa e na maioria das vagas de ML/CV.

**Custo / o que abrimos mão.** TensorFlow tem o melhor caminho para inferência
**no navegador** (TF.js) e em mobile (TFLite) — e o próprio MediaPipe é TF por
baixo. Se um dia quisermos rodar o classificador 100% client-side, a ponte é
exportar para **ONNX** e usar `onnxruntime-web`. A decisão não fecha essa porta,
mas adiciona um passo.

---

## ADR-003 — Pinagem de numpy 2.2.x (conflito ABI)

**Data:** 2026-07-14 · **Status:** aceito

**Problema.** `pip` retornou `ResolutionImpossible` ao instalar as dependências.
Lendo os metadados das wheels:

| pacote               | exigência de numpy |
| -------------------- | ------------------ |
| `mediapipe==0.10.21` | `numpy<2`          |
| `opencv-python==4.12`| `numpy>=2,<2.3`    |

Nenhuma versão satisfaz as duas.

**Decisão.** Subir o MediaPipe para **0.10.35**, que declara `numpy` sem limite
superior, e pinar **`numpy==2.2.6`** (dentro do intervalo exigido pelo OpenCV).

**Racional.** `numpy` expõe uma **ABI em C**; OpenCV e MediaPipe são compilados
*contra* ela. O `numpy 2.0` quebrou essa ABI, e o ecossistema científico levou
meses para se realinhar. A solução correta é achar a **interseção real** das
restrições — nunca instalar com `--no-deps`, que só troca um erro de instalação
por um `ImportError` binário no meio do treino.

**Custo.** Ficamos presos a `numpy < 2.3` até o OpenCV afrouxar o teto.

---

## ADR-004 — `libras/` como pacote compartilhado entre treino e inferência

**Data:** 2026-07-14 · **Status:** aceito

**Problema.** Onde mora o código de extração e normalização de landmarks?

**Decisão.** Num único pacote `libras/`, instalado em modo editável
(`pip install -e .`) e importado **tanto** pelos scripts de `training/` **quanto**
pelo `backend/`.

**Racional.** Se a função de normalização for duplicada entre o script de treino
e o servidor de inferência, um dia as duas cópias divergem. O modelo passa a
receber, em produção, features com distribuição diferente da que viu no treino.
**Ele não levanta exceção — apenas erra silenciosamente.** Isso se chama
**training/serving skew** e é o bug mais caro de diagnosticar em ML aplicado.
Um único ponto de definição torna esse bug impossível de escrever.

**Custo.** Uma camada de indireção a mais e a necessidade do `pip install -e .`
no setup.
