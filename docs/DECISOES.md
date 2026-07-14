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

---

## ADR-005 — Um único pacote de OpenCV (`opencv-contrib-python`)

**Data:** 2026-07-14 · **Status:** aceito

**Problema.** O `doctor.py` acusou `cv2.__version__ == "5.0.0"` num ambiente onde
`requirements.txt` pinava `opencv-python==4.12.0.88`. O `pip list` mostrava
**os dois** instalados:

```
opencv-contrib-python     5.0.0.93
opencv-python             4.12.0.88
```

**Causa.** `opencv-python` e `opencv-contrib-python` são pacotes distintos que
instalam **o mesmo módulo** (`site-packages/cv2/`). O pip não os trata como
conflitantes, então instala ambos — e o último a gravar no disco **sobrescreve**
os arquivos do outro. O resultado é um venv em estado misto: os metadados do pip
dizem uma coisa e o `import cv2` faz outra. O MediaPipe depende de
`opencv-contrib-python`, e foi ele quem trouxe a 5.0 por cima.

**Decisão.** Usar **exclusivamente** `opencv-contrib-python==4.12.0.88`.
Remover `opencv-python` do `requirements.txt`. Ao corrigir, desinstalar **os
dois** e reinstalar do zero — desinstalar só um deixa arquivos órfãos do outro
no diretório `cv2/`.

**Por que a série 4.x e não a 5.0.** A 5.0 é recém-lançada e removeu/renomeou
APIs. Toda a documentação, todo o Stack Overflow e todo tutorial de visão
computacional ainda assume 4.x. Num projeto cujo objetivo é aprender, **estar
alinhado com a documentação do mundo vale mais do que estar na versão mais nova.**

**Custo.** O `contrib` é ~30 MB maior (traz módulos extras que não usamos). É um
preço trivial pela integridade do ambiente — e não havia escolha, já que o
MediaPipe exige o contrib.

---

## ADR-006 — API `tasks` do MediaPipe (a `solutions` não existe mais)

**Data:** 2026-07-14 · **Status:** aceito

**Problema.** Praticamente toda a literatura de "reconhecimento de sinais com
MediaPipe" usa `mp.solutions.hands.Hands()`. No **mediapipe 0.10.35**,
`mp.solutions` **não existe** — foi removido. Verificado em ambiente:

```python
>>> import mediapipe as mp
>>> hasattr(mp, "solutions")
False
```

**Decisão.** Usar a API `mediapipe.tasks.python.vision` (`HandLandmarker`,
`PoseLandmarker`, `HolisticLandmarker`).

**Consequências.**

1. **O modelo não vem mais no pacote.** É preciso baixar um arquivo `.task` e
   apontar o caminho dele nas options. Isso vira um passo de setup
   (`scripts/download_models.py`) e um item do `.gitignore`.
2. **Ganhamos o modo `LIVE_STREAM`**: inferência assíncrona com callback,
   projetada para vídeo ao vivo — não bloqueia o loop de captura. É o modo
   correto para o nosso caso de uso.
3. **Ganhamos um caminho para o navegador**: o mesmo arquivo `.task` roda em
   JS via `@mediapipe/tasks-vision`. Isso mantém aberta a otimização da Etapa 16
   (extrair os landmarks no cliente e mandar só ~126 floats pro backend, em vez
   de um JPEG por frame).
4. **Nenhum tutorial de terceiros vai copiar-e-colar.** É uma feature, não um
   bug: a fonte da verdade passa a ser a API instalada, não um vídeo de 2023.

**Custo.** API mais verbosa e um asset externo para gerenciar.

---

## ADR-007 — Backend de captura MSMF (escolhido por medição, não por intuição)

**Data:** 2026-07-14 · **Status:** aceito

**Problema.** Qual backend do OpenCV usar para abrir a webcam no Windows?

**O que eu decidi primeiro, e errado.** `cv2.CAP_DSHOW` (DirectShow), com este
comentário confiante no código: *"no Windows, o backend padrão (MSMF) costuma
levar ~2s para abrir a câmera; DirectShow abre instantaneamente"*. O comentário
está **factualmente correto**. A decisão está **errada**.

**O que a medição mostrou** (`scripts/bench_camera.py`, 640×480, nesta máquina):

| backend | FPS real | ms/frame | codec |
| ------- | -------- | -------- | ----- |
| **MSMF**  | **30.0** | **33.4** | (negociado) |
| DSHOW   | 15.0     | 66.6     | YUY2  |
| ANY     | 25.0     | 40.1     | —     |

O DirectShow ficava preso em **YUY2** — vídeo **não comprimido**, ~614 KB por
frame. A 30 FPS isso seriam ~18 MB/s no barramento USB; o driver, para não
estourar a banda, **corta a taxa pela metade**. E ele ignorava tanto o pedido de
30 FPS quanto a troca para MJPG: aceitava a chamada e continuava entregando 15.

**Decisão.** `cv2.CAP_MSMF` como padrão (`CameraConfig.backend`). Em Linux/Mac,
`cv2.CAP_ANY`.

**Racional.** Otimizei a coisa errada. **Abrir a câmera acontece uma vez; ler
frames acontece 30 vezes por segundo, para sempre.** Trocar 2 segundos de
inicialização por metade do FPS — permanentemente — é um péssimo negócio.

**A lição que fica.** Eu tinha um motivo plausível, escrito com convicção num
comentário, e ele me levou à escolha errada. O que corrigiu isso não foi
raciocínio melhor: foi **medir**. É exatamente por isso que o `Cronometro`
existe *antes* de o MediaPipe entrar no loop. E é por isso que
`scripts/bench_camera.py` está versionado: em outra máquina o vencedor pode ser
outro, e a resposta certa é rodar de novo — não confiar neste ADR.

---

## ADR-008 — Modelo de custo do loop: `max()`, não soma

**Data:** 2026-07-14 · **Status:** aceito (é um fato medido, não uma escolha)

**O erro de raciocínio.** É natural pensar que o tempo de um frame é
`captura + mediapipe + modelo`, e concluir que, se a captura já custa 33 ms, não
sobra nada para a IA. **Errado.**

`cap.read()` não *gasta* CPU: ele fica **bloqueado esperando** a câmera produzir
o próximo frame. Se o processamento levar 20 ms, a câmera trabalhou *durante*
esses 20 ms, e o `read()` seguinte retorna quase instantâneo. Com
`CAP_PROP_BUFFERSIZE = 1` (descartar frames velhos), o loop se estabiliza em:

```
tempo_do_frame  ≈  max( período_da_câmera , tempo_de_processamento )
```

**Medido** (câmera a 30 FPS ⇒ período de 33,3 ms; processamento simulado
queimando CPU):

| processamento | FPS medido |
| ------------- | ---------- |
| 0 ms          | 29.6       |
| 10 ms         | 29.5       |
| 20 ms         | 29.5       |
| 30 ms         | 29.5       |
| 40 ms         | 20.0       |
| 60 ms         | 13.4       |

**Consequências para o projeto.**

1. **O MediaPipe tem ~33 ms de folga, e usá-los é grátis.** Enquanto o
   processamento couber no período da câmera, o FPS não cai *nada*.
2. **A degradação não é suave — ela quantiza.** Passando de 33 ms, você perde o
   "trem" do frame e espera o próximo inteiro. Por isso 40 ms derrubou para 20
   FPS (e não para os 25 que uma regra de três ingênua previa). Fique
   confortavelmente **abaixo** do limite, não em cima da linha.
3. **Se um dia o processamento passar de 33 ms**, a saída não é otimizar o
   modelo às cegas: é **desacoplar** captura e inferência em threads, ou usar o
   modo `LIVE_STREAM` do MediaPipe (que é assíncrono justamente para isto).
   Fica para a Etapa 16.

---

## ADR-009 — Modo `VIDEO` do HandLandmarker (e por que isso é rápido)

**Data:** 2026-07-14 · **Status:** aceito

**O que é preciso entender primeiro.** O `HandLandmarker` **não é um modelo, são
dois**:

1. **palm detector** — varre o frame **inteiro** procurando "onde tem mão?". Caro.
2. **landmark model** — recebe um *recorte* que já contém a mão e devolve os 21
   pontos. Barato.

Rodar os dois em todo frame seria lento demais para tempo real. O MediaPipe roda
o detector **uma vez**, acha a mão, e nos frames seguintes usa a posição anterior
para recortar a região e rodar **só o modelo barato**. Ele é um **tracker**. Só
volta a chamar o detector caro quando **perde** o rastreamento.

**Decisão.** Usar `running_mode=VIDEO` (mantém estado entre frames ⇒ o tracker
funciona), e **não** `IMAGE` (trata cada frame como foto isolada ⇒ o detector
caro roda **sempre**).

**Os dois parâmetros de confiança, que sempre confundem:**

| parâmetro | quando age | significado |
| --------- | ---------- | ----------- |
| `min_hand_detection_confidence` | quando o **detector** roda | quão certo preciso estar de que isto é uma mão para começar a rastrear |
| `min_tracking_confidence` | em **todo frame rastreado** | abaixo disto eu desisto e chamo o detector caro de novo |

**Armadilhas encontradas na prática:**

1. **Timestamps.** O modo `VIDEO` exige timestamps **estritamente crescentes**.
   Repetir ou regredir um levanta exceção com mensagem inútil. Por isso o
   `DetectorMaos` mantém o próprio contador em vez de confiar em quem chama.
2. **Cold start.** A primeira inferência carrega os pesos, aloca buffers nativos
   e inicializa o delegate XNNPACK — custa **várias vezes** o normal. Isso
   envenenou a primeira versão do `bench_maos.py` (primeiro cenário mediu 55 ms
   contra 16 ms dos demais: era cold start disfarçado de resultado). O
   aquecimento agora acontece dentro do `DetectorMaos.abrir()`, para que ninguém
   possa esquecer.
3. **Benchmark sem mão no quadro não mede nada.** Sem mão, o tracker não tem o
   que rastrear ⇒ o modo `VIDEO` também roda o detector em todo frame ⇒ os dois
   modos fazem **o mesmo trabalho**. O `bench_maos.py` agora se **recusa a
   concluir** se a mão aparecer em menos de 80% dos frames. Um benchmark que
   produz um número mesmo quando a premissa não vale é pior que benchmark nenhum:
   alguém vai citar esse número depois.
4. **Os avisos `W0000` do MediaPipe não são suprimíveis por variável de
   ambiente.** A tentativa óbvia (`GLOG_minloglevel=2` antes do import) **não
   funciona**: o MediaPipe migrou de `glog` para o logging do `absl`, que ignora
   essa variável. E `contextlib.redirect_stderr` também não pega — ele só troca
   o objeto `sys.stderr` do **Python**, enquanto o C++ escreve **direto no file
   descriptor 2**. A solução real é `os.dup2` (nível de SO), aplicada **apenas**
   durante a abertura do modelo — silenciar o stderr permanentemente esconderia
   erros de verdade.

---

## ADR-010 — `hand_world_landmarks` como base das features

**Data:** 2026-07-14 · **Status:** aceito

**Problema.** O MediaPipe devolve os 21 pontos em **dois espaços de coordenadas**.
Qual usar como feature do classificador?

| campo | espaço | bom para |
| ----- | ------ | -------- |
| `hand_landmarks` | normalizado 0..1 **pela imagem** | **desenhar** (multiplica por largura/altura ⇒ pixel) |
| `hand_world_landmarks` | **métrico 3D, em metros**, origem no centro da mão | **features** |

**Decisão.** `hand_world_landmarks` alimenta o modelo; `hand_landmarks` só
desenha.

**Racional.** Os landmarks normalizados pela imagem mudam **todos** quando você
anda para o lado ou se aproxima da câmera — embora o sinal seja exatamente o
mesmo. Um classificador treinado neles teria que *aprender* a ignorar posição e
escala, e para isso precisaria de muito mais dados.

Os `world landmarks` já vêm **invariantes a translação e a escala**: mesma mão,
mesma pose, números praticamente iguais, independentemente de onde ela esteja no
quadro. Metade do trabalho de *feature engineering* vem pronta — **de graça**,
por usarmos um modelo pré-treinado. Isso é o ADR-001 pagando dividendos.

**O que ainda falta** (fica para a Etapa 7): eles **não** são invariantes a
**rotação**. Inclinar a mão muda os números. Vamos ter que decidir se
normalizamos a rotação ou se deixamos o modelo aprender a lidar com ela.

---

## ADR-011 — Corrigir a lateralidade **na fronteira** (revoga uma decisão anterior)

**Data:** 2026-07-14 · **Status:** aceito · **Revoga:** a nota "não vamos
consertar" que estava em `Mao.lado`

**O sintoma.** O preview mostrava **"Left" sobre a mão direita** do usuário, e
vice-versa.

**A causa.** Espelhamos o frame na captura (senão sinalizar é desorientador —
você levanta a direita e ela aparece à esquerda da tela). O MediaPipe recebe o
frame **já espelhado** e classifica a lateralidade **do que ele vê**: a mão
direita, espelhada, tem a geometria de uma mão esquerda. Ele responde "Left", e
está sendo **coerente com a imagem que recebeu**.

**O que eu tinha decidido, e errado.** Deixar como estava, com este comentário
no código: *"não é um bug e não vamos consertar: é uma convenção, e o que importa
é ela ser a mesma na coleta, no treino e na inferência; trocar os rótulos só aqui
é criar duas convenções e escolher a errada em algum lugar."*

A **premissa** está certa — duas convenções concorrentes é desastre. A
**conclusão** não. Eu só enxerguei duas opções (deixar errado *ou* remendar na
exibição) e não vi a terceira.

**A decisão.** Corrigir **na fronteira**: no `DetectorMaos._converter`, o único
ponto por onde os dados do MediaPipe entram no sistema. Assim continua havendo
**uma só convenção** — e agora ela é **verdadeira**.

**Detalhe que faz a coisa toda funcionar:** trocamos o **rótulo**, nunca a
**geometria**. Os landmarks continuam vindo da imagem espelhada. A mesma mão real
produz sempre a mesma geometria **e** o mesmo rótulo, na coleta e na inferência.
A consistência treino/serving está preservada — que era a preocupação original.

**Salvaguardas.**

- `DetectorMaos(entrada_espelhada=...)` é **explícito** e, nos scripts, é
  **derivado** de `CameraConfig.espelhar` — não repetido à mão. Se alguém
  desligasse o espelhamento e esquecesse de ajustar o detector, os rótulos
  sairiam trocados **sem erro nenhum**, apenas contaminando o dataset.
- `Mao.lado_bruto` guarda a resposta original do MediaPipe. Se um dia os rótulos
  parecerem trocados, comparar `lado` com `lado_bruto` diz na hora se o problema
  é o espelho ou o modelo — diagnóstico em 10 segundos, não em 2 dias.
- Cinco testes (`TestLateralidade`) travam o comportamento.

**A lição.** Um campo chamado `lado` que diz "Left" para a mão direita é uma
armadilha esperando alguém — inclusive o você de daqui a três meses, montando o
dataset. **"É uma convenção" não é desculpa para um nome mentir.** E, mais
importante: quando as duas opções óbvias são ruins, procure a terceira.
