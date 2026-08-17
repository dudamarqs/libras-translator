# Tradutor de Libras em Tempo Real

Reconhecimento de datilologia (alfabeto manual de Libras) pela webcam, traduzido
para texto em tempo real — com a confiança da predição exibida junto, e com o
sistema **se recusando a responder** quando não reconhece o que vê.

> **Status:** em construção — Etapa 11/19. O circuito completo já funciona ao vivo:
> câmera → landmarks → features → classificador → montador de palavras → corretor
> → legenda em português na tela.
> Modelo atual reconhece as **20 letras estáticas** do alfabeto
> (A B C D E F G I L M N O P Q R S T U V W), treinado com 9.271 amostras
> em 5 sessões de captura.

<!-- TODO: GIF do demo ao vivo aqui (letras sendo reconhecidas + o "?" ao coçar a cabeça) -->

---

## Como o sistema funciona

```
  webcam  ──►  MediaPipe  ──►  normalização  ──►  classificador  ──►  "C" (0.98)
  (frame)      (landmarks)     (vetor 63)         (LogReg)             letra + confiança
                                                       │                     │
                                                       ▼                     ▼
                                              detecção de novidade      montador
                                              (longe do treino? → "?")   (letras → palavra)
                                                                             │
                                                                             ▼
                                                                    corretor (thread)
                                                                    "CAURO" → "carro"
```

O ponto central da arquitetura: **não classificamos pixels.** O MediaPipe — uma
rede pré-treinada pelo Google — converte cada frame nas coordenadas 3D das
articulações das mãos. O frame de 921.600 bytes vira um vetor de **63 números**
(21 pontos × 3 eixos) que descreve apenas a **geometria** do gesto, já livre de
fundo, iluminação e aparência do sinalizante.

Isso é *transfer learning*, e é o que torna o projeto viável: bastam ~100–200
amostras por sinal, e o modelo treina em segundos numa CPU.

O raciocínio completo — inclusive o que foi descartado e por quê — está em
[docs/DECISOES.md](docs/DECISOES.md) (21 ADRs).

---

## O problema mais interessante que este projeto resolve

Um classificador de **conjunto fechado** conhece N formas de mão e nada além.
Dada qualquer mão, ele é *obrigado* a escolher uma das N — não existe saída
"nenhuma delas".

No primeiro teste ao vivo, o sistema viu uma mão **coçando a cabeça** e cravou
`C` com **100% de confiança**. E não é bug de treino: num modelo linear, quanto
mais longe da fronteira de decisão o ponto cai, **mais confiante** fica a
softmax. Aumentar o limiar de confiança não resolve — o modelo está 100%
confiante no erro.

A solução foram duas defesas independentes:

| defesa | o que ataca |
| ------ | ----------- |
| **Detecção de novidade** — distância aos K vizinhos de treino mais próximos; longe demais ⇒ `desconhecido` | o chute confiante em mãos que não são letras |
| **Trava de imobilidade** — só classifica com a *forma* da mão parada | letras capturadas no meio da transição |

O limiar de novidade não é número mágico: sai da própria distribuição dos dados
(percentil 99 das distâncias intra-treino). Verificação: **0%** de letras reais
barradas, **100%** de geometria inválida barrada.

Detalhes em [ADR-016](docs/DECISOES.md).

---

## Resultados

Validação por **`LeaveOneGroupOut` sobre as sessões de coleta** — treina em N−1
sessões, testa na que sobrou. Cada dobra responde: *"treinei em duas condições de
luz, acerto numa terceira que nunca vi?"*

*(20 letras estáticas, 9.271 amostras, 5 sessões em luzes e dias diferentes.)*

**O número honesto é 88,4%.** A média das 5 dobras é 95,9%, mas ela é enganosa:
três dessas dobras são sessões antigas que contêm apenas A, B, C, L e O — cinco
letras fáceis, onde o modelo acerta 98–100% e infla a média. Só duas dobras
testam o alfabeto inteiro:

| sessão de teste | letras na dobra | acurácia |
| --------------- | --------------- | -------- |
| 20260715 | 5 | 98,4% |
| 20260716 | 5 | 100% |
| 20260717 | 5 | 99,7% |
| **20260723** | **20** | **93,1%** |
| **20260728** | **20** | **88,4%** |

Comparação entre modelos sob a mesma validação:

| modelo | acurácia (média LOGO) | observação |
| ------ | --------------------- | ---------- |
| **LogisticRegression** | **95,9%** | modelo de produção |
| RandomForest | 93,2% | mais instável entre sessões |
| MLP (PyTorch) | 96,0% | empata na média e **perde** na sessão difícil (87,8% vs 88,4%) |

> ⚠️ **Ressalva honesta.** As sessões são da **mesma pessoa e da mesma mão**.
> O número prova generalização entre **luz e posição**, não entre **pessoas**.
> Uma mão com proporções diferentes é território que o modelo nunca viu.

### A rede neural foi construída — e perdeu

A MLP (63→128→64→20) chegou a 100% no treino e 87,8% no teste: **12,3 pontos de
gap**, decorou sem generalizar. O modelo linear continua em produção.

Isso responde a pergunta que interessa: se a capacidade do modelo fosse o
gargalo, a rede — que provou ter capacidade de sobra para decorar o treino —
teria generalizado melhor. Não teve. **O gargalo são os dados**, especificamente
a consistência da sinalização entre sessões nos clusters R↔U↔V e T↔F.

O diagnóstico confirmou isso: R/U e T/F são 99–100% separáveis *dentro* de uma
sessão e caem para 55–80% *entre* sessões. E a tentativa de consertar com
augmentation de rotação **piorou** o resultado (92,5% → 90,9%), porque R, U e V
se distinguem justamente pela orientação dos dedos — ensinar invariância a
rotação apaga o sinal que separa as classes.

Ver [ADR-002](docs/DECISOES.md), [ADR-015](docs/DECISOES.md),
[ADR-017](docs/DECISOES.md) e [ADR-018](docs/DECISOES.md).

---

## Rodando

Depois do [Setup](#setup):

```powershell
# 1. coletar o dataset (uma sessão = uma condição de luz, alfabeto inteiro)
python training\collect.py

#    regravar letras específicas de uma sessão já salva, sem refazer o resto
python training\collect.py --corrigir sessao_20260721_232645 --letras "A B C"

# 2. cru -> features
python training\preprocess.py

# 3. baseline honesto (LeaveOneGroupOut + matriz de confusão)
python training\train.py

# 4. treinar o modelo final em tudo e exportar para produção
python training\exportar.py

# 5. o demo ao vivo (webcam -> letras -> palavras -> legenda corrigida)
python scripts\reconhecer.py

# 5b. a camada de texto sem câmera ("CAURO" -> "carro"), para ver os dois corretores
python scripts\demo_legenda.py
```

A correção **não roda no loop de vídeo**. Um frame dura 33 ms e uma correção
custa 582 ms no modo offline (ou 1–3 s via LLM): chamar direto congelaria o vídeo
por 17 a 90 frames. Ela roda numa thread, com slot de um pedido — último vence —
e um contador de geração que invalida resultado em voo, senão a legenda
"ressuscita" texto que o usuário já apagou. Custo medido no loop: **0,03 ms**.
Ver [ADR-021](docs/DECISOES.md).

No demo, cada frame cai num de quatro estados — e **só o verde digita**:

| estado | quando |
| ------ | ------ |
| `desconhecido` (`?`) | mão longe de tudo que o modelo viu |
| `movendo...` | a forma da mão ainda está mudando |
| `incerto` | parada, mas confiança abaixo do limiar |
| `ok` | reconhecida + parada + confiante + estável |

O HUD mostra `mov:` e `dist:` — os números crus por trás dessas decisões, para
recalibrar os limiares olhando dados em vez de chutando.

---

## Estrutura

| Pasta | Função |
| ----- | ------ |
| `libras/` | **Pacote compartilhado.** Extração e normalização de landmarks, dataset, classificador e o catálogo de sinais. Importado *tanto* pelo treino *quanto* pela inferência — é isso que impede o `training/serving skew` ([ADR-004](docs/DECISOES.md)). |
| `training/` | Scripts do ciclo de ML: `collect.py`, `preprocess.py`, `train.py`, `exportar.py`. |
| `scripts/` | Executáveis de uso: `reconhecer.py` (demo ao vivo), `doctor.py`, previews e benchmarks. |
| `tests/` | 104 testes (pytest), incluindo testes de concorrência determinísticos (sem `sleep`). |
| `docs/` | `DECISOES.md` — 21 ADRs com o porquê de cada escolha. |
| `datasets/` | `raw/` (o que a webcam capturou) e `processed/` (features). **Fora do Git.** |
| `models/` | Modelo treinado + `meta.json` (classes, métricas, commit). **Fora do Git.** |
| `backend/` | *Vazio — planejado.* API FastAPI com WebSocket de inferência. |
| `frontend/` | *Vazio — planejado.* Interface React + Vite + Tailwind. |
| `notebooks/` | *Vazio — planejado.* Experimentação exploratória. |

`datasets/` e `models/` ficam **fora do Git**: o Git guarda a história completa
de cada arquivo, então um dataset regravado 10 vezes vira um repo gigante — para
sempre, mesmo depois de você deletar o arquivo.

---

## Setup

Requer **Python 3.12** (o MediaPipe não publica wheels para 3.13/3.14).

```powershell
# 1. venv com o Python 3.12 explicitamente
C:\Users\eduar\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv

# 2. ativar
.venv\Scripts\Activate.ps1

# 3. dependências (use -dev para desenvolver; requirements.txt sozinho é o runtime)
python -m pip install -r requirements-dev.txt

# 4. tornar o pacote `libras` importável de qualquer lugar
python -m pip install -e .

# 5. baixar o modelo do MediaPipe
python scripts\download_models.py

# 6. verificar que tudo realmente funciona
python scripts\doctor.py
```

O `doctor.py` valida o que `pip install` **não** valida: se numpy/OpenCV/MediaPipe
trocam arrays sem estourar a ABI em C, se a webcam entrega frame de verdade, e
se o pacote `libras` está importável.

---

## Escopo

**O que está pronto:** datilologia **estática** — as letras que cabem em um único
frame. Reconhecimento de uma mão, em tempo real, com abstenção.

**O que também está pronto:** a camada de texto — o `montador` junta as letras em
palavras e frases (lógica pura, sem IA), e o `corretor` conserta o resultado.
São duas implementações atrás da mesma interface: uma por LLM (Claude) e uma
**offline e gratuita** (dicionário + distância de edição). A fábrica escolhe a
melhor disponível, então o projeto roda de graça por padrão e o LLM é upgrade
opcional — e o offline vira fallback quando a API falha.

**O que não está:** as letras **com movimento** (H, J, K, X, Y, Z) exigem um
modelo temporal (LSTM) e ficam para a fase dinâmica — estão declaradas como
`dinamico` em [`sinais.yaml`](libras/registry/sinais.yaml) e o coletor as pula
automaticamente. Sinais de duas mãos também são fase seguinte.

Não existe modelo de Libras pronto para usar: os modelos públicos são de **ASL**,
que é outra língua. O MediaPipe entrega landmarks, não significado — o
significado vem dos dados que você rotula.

---

## Stack

**Visão computacional:** OpenCV (captura) + MediaPipe (landmarks).
**ML:** scikit-learn (baseline e modelo de produção) + PyTorch (MLP avaliada em
[ADR-017](docs/DECISOES.md) — construída, medida e descartada).
**Camada de texto:** montador puro + corretor por Claude (`anthropic`) ou offline
(`pyspellchecker`), atrás da mesma interface.
**Qualidade:** pytest + ruff.
**Previsto:** FastAPI + Uvicorn (WebSocket a 30 FPS) e React + Vite + Tailwind.
