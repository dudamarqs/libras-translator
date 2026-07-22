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

---

## ADR-012 — Normalizar translação e escala; **preservar a rotação**

**Data:** 2026-07-14 · **Status:** aceito

**Problema.** O `hand_world_landmarks` (ADR-010) já resolve a **posição no
quadro**. Sobram duas fontes de variação que não têm nada a ver com o sinal:

1. **Escala.** Os valores vêm em **metros reais**. Uma criança tem a mão ~30%
   menor: mesmo sinal, todos os números 30% menores — e o modelo veria dois
   gestos diferentes.
2. **Rotação.** Inclinar a mão muda todos os números.

**Decisão.**

| transformação | o que fazemos | por quê |
| ------------- | ------------- | ------- |
| **translação** | recentrar no **pulso** | o "centro geométrico" do MediaPipe é uma média dos 21 pontos e **se move quando os dedos dobram**. Um referencial que se mexe não é referencial. O pulso é uma **âncora anatômica**. |
| **escala** | dividir pela distância **pulso → base do dedo médio** | torna a mão **adimensional**: importa a *proporção* entre os pontos, não o tamanho |
| **rotação** | **NÃO normalizar** | ver abaixo |

**Por que a régua é pulso → base do médio, e não pulso → ponta do médio.**
A régua tem que depender **só do tamanho da mão, nunca do gesto**. Pulso → base
do médio atravessa a **palma**, que é rígida: mede o mesmo com o punho fechado
ou a mão aberta. Já pulso → **ponta** do médio varia de ~18 cm (mão aberta) a
~8 cm (punho fechado) — depende do **gesto**. Usá-la faria o punho fechado
parecer uma mão aberta gigante. *Uma régua não pode depender daquilo que ela
mede.*

**Por que NÃO normalizamos a rotação — e esta é a decisão que importa.**

A saída óbvia seria girar a mão para uma orientação canônica. Seria matematicamente
elegante e **destruiria o projeto**.

Um sinal em Libras tem **cinco parâmetros**: configuração de mão, ponto de
articulação, movimento, **orientação da palma** e expressão facial. A orientação
da palma é **gramática** — palma para dentro e palma para fora podem ser sinais
**diferentes**. Normalizar a rotação apagaria **um dos cinco parâmetros da
língua**, e o modelo ficaria estruturalmente cego para distinções que os
sinalizantes usam todo dia.

> A resposta certa aqui não veio da matemática. Veio de **entender o domínio**.
> É o tipo de decisão que separa um projeto de ML de portfólio de um projeto de
> ML de verdade — e é uma ótima resposta para "conte uma decisão técnica difícil
> que você tomou".

**Salvaguarda.** Existe um teste chamado
`test_NAO_e_invariante_a_ROTACAO__e_isso_e_intencional`. Ele parece testar um
defeito; não é — ele **trava a decisão**. Se alguém um dia "melhorar" a
normalização acrescentando invariância a rotação (parece um upgrade óbvio!), o
teste quebra e obriga a pessoa a ler este ADR antes de apagar um parâmetro da
língua. *Um teste que protege uma decisão vale mais que um teste que confere uma
conta.*

**Custo.** O modelo terá que aprender a lidar com pequenas variações de
inclinação sozinho — e para isso precisa vê-las no dataset. Consequência direta
para a Etapa 6: **ao coletar, varie levemente a inclinação da mão**, ou o modelo
só reconhecerá o sinal no ângulo exato em que foi gravado.

**Layout final do vetor de entrada** (`vetor_features`, 128 dimensões):

```
[  0.. 62]  mão esquerda (21 pontos × 3 eixos, normalizados)
[ 63..125]  mão direita  (idem)
[126]       flag: mão esquerda presente? (0.0 / 1.0)
[127]       flag: mão direita presente?  (0.0 / 1.0)
```

Três decisões embutidas aí:

1. **Posição fixa por lado**, não por ordem de detecção. O MediaPipe **não
   garante a ordem** das mãos, e ela muda entre frames — o mesmo sinal geraria
   vetores diferentes. (Isto só é possível porque corrigimos a lateralidade no
   ADR-011.)
2. **Mão ausente = zeros.** Rede neural tem entrada de tamanho fixo.
3. **As flags de presença existem por causa de (2)** — e são a parte que quase
   todo mundo esquece. Sem elas, o modelo não consegue distinguir *"esta mão não
   está no quadro"* de *"esta mão está numa pose cujos números deram perto de
   zero"*. São situações muito diferentes que o vetor representaria de forma
   parecida.

---

## ADR-013 — Formato do dataset: landmarks crus + agrupamento por sessão

**Data:** 2026-07-15 · **Status:** aceito

**Problema.** Como armazenar o dataset coletado da datilologia?

**Três decisões, cada uma evitando um erro clássico.**

**1. Guardar landmarks CRUS, não as features já normalizadas.**
Coletar de novo exige o humano na frente da câmera — é caro. Recalcular features
é grátis. Se a `features.py` evoluir (e ela vai — falta tratar rotação de forma
mais esperta, decidir sobre a mão dominante etc.), com os dados crus a gente
**reprocessa**; se tivéssemos salvo só as features, teríamos que **recoletar
tudo**. Regra: *guarde perto do cru, derive as features na hora.* O
pré-processamento (Etapa 7) aplica `features.py` sobre o cru — reusando a mesma
função da inferência, sem risco de skew (ADR-004).

**2. Cada sessão de gravação é um GRUPO, e isso fica gravado em cada amostra.**
A 30 FPS, frames vizinhos são quase idênticos. Um split treino/teste aleatório
colocaria cópias quase perfeitas nos dois lados — o teste viraria uma cópia do
treino, e a acurácia medida seria uma mentira (*data leakage*). A defesa: o
split é feito por **sessão inteira** (nenhuma sessão nos dois lados). Para isso,
`DatasetBruto.grupos[i]` guarda o `session_id` da amostra `i`. Um teste
(`test_cada_amostra_conhece_a_propria_sessao`) protege esse rastreamento.

Consequência prática **para você**: faça **várias sessões**, em dias e luzes
diferentes. É a variedade *entre* sessões que ensina o modelo a generalizar —
não a quantidade de frames de uma sessão só.

**3. Formato inspecionável e autodescritivo.** Cada sessão é uma pasta com
`dados.npz` (arrays comprimidos) + `meta.json` (quando, qual câmera, qual commit
de código gerou os dados). Isso é *data provenance*: daqui a seis meses você
ainda sabe como cada amostra nasceu, e pode descartar uma sessão específica se
descobrir que ela estava ruim.

**O registro de sinais (`sinais.yaml`).** O catálogo de sinais é declarativo:
adicionar um sinal é adicionar uma linha no YAML, lido pelo coletor, pelo treino
e pela inferência. É o ADR-004 (fonte única) aplicado aos **metadados**. Ele
também carrega a distinção `estatico`/`dinamico` — que é o que permite o coletor
pular automaticamente as letras com movimento (H, J, K, X, Y, Z) na fase atual.

**Custo.** Um passo de pré-processamento a mais (cru → features) antes de cada
treino. É barato, e compramos com ele a liberdade de mudar as features sem
recoletar — uma troca excelente.

---

## ADR-014 — Datilologia usa o vetor de UMA mão (63d), não o de 128

**Data:** 2026-07-15 · **Status:** aceito · **Descoberto por:** dados reais

**O que os dados revelaram.** Na primeira coleta real (letras A B C L O, mão
direita), o MediaPipe classificou **14 de 805 frames como "Left"** — 12 no `C`,
2 no `O`. São letras de mão "redonda", quase simétrica, e o classificador de
lateralidade do MediaPipe se confunde com elas (~1,7% dos frames).

**Por que isso é um problema para o vetor de 128.** O `vetor_features` (ADR-012)
ancora cada mão no bloco do seu lado (esquerda 0..62, direita 63..125). Com o
lado errado em 14 frames, esses cairiam no bloco **errado** — o modelo veria a
mesma letra `C` em duas regiões diferentes do vetor, com só 12 exemplos na
região "errada". Pior: em produção, um flip de lateralidade do MediaPipe jogaria
a predição para o bloco vazio e o modelo falharia.

**Decisão.** Para sinais de **uma mão** (toda a datilologia), usar
`vetor_uma_mao` → **63 dimensões**, a geometria da mão detectada, **ignorando o
lado**. `normalizar_mao` não olha o lado nenhuma vez (centra no pulso, escala),
então os 14 frames com lado errado têm features **perfeitamente válidas** — nada
se perde. Robusto ao erro de lateralidade **por construção**.

O vetor de 128 (com âncora por lado) continua correto e será usado quando
chegarmos a sinais de **duas mãos**. Cada representação para o seu tipo de sinal.

**A lição.** A âncora-por-lado parecia uma vantagem no papel (ADR-012), e é —
para duas mãos. Foram os **dados reais** que mostraram que, para uma mão, ela é
uma *desvantagem*. Nenhum raciocínio de mesa tinha pego isso; a coleta pegou.
Mais uma vez: medir, não supor.

**Custo.** Um modelo treinado com a mão direita vê a mão esquerda **espelhada** e
não a reconhece. Mitigação (Etapa 8): *data augmentation* espelhando o eixo x
sintetiza a outra mão de graça — dobra o dataset e cobre canhotos e destros.

---

## ADR-015 — `LeaveOneGroupOut` como validação (e não `GroupShuffleSplit`)

**Data:** 2026-07-17 · **Status:** aceito

**Problema.** Com o dataset agrupado por sessão (ADR-013), como medir a acurácia
de forma honesta e **reprodutível**?

**O que estava errado antes.** A primeira medição de separabilidade deu
*LogReg 99,8% / RandomForest 100%* — números lindos e **mentirosos**: eram um
5-fold aleatório dentro de **uma única sessão**. A 30 FPS, frames vizinhos são
quase gêmeos, então treino e teste continham cópias um do outro. Aquilo media
memorização, não generalização.

**Decisão.** Validar com `LeaveOneGroupOut` sobre `grupos` (o `session_id`):
com N sessões, N dobras, e cada sessão é o conjunto de teste **exatamente uma
vez**, treinando nas demais. Cada dobra responde à pergunta que interessa de
verdade: *"treinei em 2 condições de luz — acerto numa terceira que nunca vi?"*

**Por que não `GroupShuffleSplit`** (que era o previsto no roteiro): ele
**sorteia** quais sessões vão para o teste. Com poucas sessões isso significa
(a) um número que muda a cada execução, e (b) sessões que nunca são testadas. O
`LeaveOneGroupOut` é **determinístico** e usa cada sessão como teste — logo, é
reprodutível e comparável entre execuções. Com poucos grupos, sortear não traz
vantagem nenhuma e custa reprodutibilidade.

**Corolário que quase passou batido — o scaler mora DENTRO do pipeline.** O
`StandardScaler` é ajustado em cada dobra de treino, nunca vendo o teste. Se
fosse aplicado uma vez sobre o `X` inteiro antes do split, a média e o desvio
carregariam informação do conjunto de teste — um vazamento sutil, invisível, que
infla o resultado sem nenhum sintoma.

**Resultado.** LogReg **99,7%** (desvio 0,1% entre sessões) contra RandomForest
97,1% (desvio 2,1%). Repare que o modelo **mais simples generalizou melhor** —
o RandomForest degradou justamente na sessão que nunca viu (94,1%).

**Consequência para o roteiro.** O baseline em 99,7% deixa ~zero espaço para uma
rede neural melhorar em 5 classes linearmente separáveis. Construir a MLP ali
seria seguir o roteiro **traindo o princípio que o roteiro codifica** (ADR-002:
o baseline decide se a rede vale). A MLP fica para quando o alfabeto crescer e a
confusão entre punhos parecidos (M/N/S/T) tornar o problema não-linear.

**Ressalva honesta e permanente.** As sessões são da **mesma pessoa e da mesma
mão**. O número prova generalização entre **luz e posição**, não entre
**pessoas**. Não anunciar "99,7%" sem essa frase junto.

---

## ADR-016 — Abstenção: o modelo precisa poder dizer "não sei"

**Data:** 2026-07-17 · **Status:** aceito · **Descoberto por:** teste ao vivo

**O que aconteceu.** Com o loop fechado funcionando, o primeiro teste real na
webcam: a usuária **coçou a cabeça** — dedos espalhados, nada parecido com
nenhuma letra — e o sistema cravou **`C` com 100% de confiança**. O texto
acumulado virou `AAABCBABCOLCABO`.

**Por que 100%, e por que subir o limiar NÃO resolve.** Um classificador de
**conjunto fechado** conhece 5 formas de mão e nada além. Dada qualquer mão, ele
é *obrigado* a escolher entre as 5 — não existe saída "nenhuma delas". Pior: num
modelo linear, quanto **mais longe** da fronteira de decisão o ponto cai, **mais
confiante** fica a softmax. A mão coçando estava muito fora de tudo que o modelo
viu, e por acaso caiu do lado do `C`. Os 100% não significam "tenho certeza que é
C"; significam "isto está longuíssimo da fronteira". **Confiança de softmax não
mede estranheza** — por isso nenhum limiar de confiança conserta isso.

**Decisão — duas defesas independentes, para dois problemas diferentes.**

**1. Detecção de novidade (`libras/models/classificador.py`).** Um
`NearestNeighbors` ajustado sobre os dados de treino **já escalados**. Na
inferência, medimos a distância média aos K=5 vizinhos de treino mais próximos:
perto ⇒ é uma das letras; longe ⇒ `desconhecido`, e o sistema **se recusa a
responder**. O limiar não é número mágico — sai da própria distribuição dos
dados (percentil 99 das distâncias intra-treino, com folga de 1,5×). É a peça
que devolve ao sistema o direito de dizer "não sei", que a softmax nunca teve.

**2. Trava de imobilidade (`scripts/reconhecer.py`).** Uma letra é um sinal
*sustentado*; coçar a cabeça e trocar de letra são *movimento*. Só classificamos
quando a **forma** da mão está parada — medida pela variação do vetor de features
entre frames (invariante a posição: arrastar a mão parada pela tela não conta
como movimento, só mudar os dedos conta). Isso resolve um sintoma diferente: a
captura de letras no meio da transição.

**Verificação.** Offline: 0% de letras reais barradas (nenhum falso
"desconhecido") e 100% de geometria embaralhada barrada. Ao vivo: coçar a cabeça
passou a exibir `?` vermelho, e as letras reais continuaram registrando fácil.

**Custo.** O artefato salvo cresce (guarda os dados de treino escalados para o
kNN) e há uma busca por vizinhos a cada frame — irrelevante nesta escala. O
limiar foi calibrado contra geometria sintética, não contra poses OOD reais; o
HUD exibe `dist:` justamente para permitir recalibrar com dados de verdade.

**A lição.** A matriz de confusão nunca mostraria isso — ela só testa com A, B,
C, L e O. O mundo real tem cabeça para coçar. **Só o teste ao vivo revela o
comportamento fora da distribuição**, e um sistema que sabe recusar vale mais
que um que acerta sempre no dataset.
