# Tradutor de Libras em Tempo Real

Reconhecimento de datilologia (alfabeto manual de Libras) pela webcam, traduzido
para texto em tempo real — com a confiança da predição exibida junto, e com o
sistema **se recusando a responder** quando não reconhece o que vê.

> **Status:** em construção — Etapa 8/19. O loop completo já funciona ao vivo:
> câmera → landmarks → features → classificador → texto na tela.
> Modelo atual reconhece **A, B, C, L, O**; a coleta do alfabeto de 20 letras
> está em andamento.

<!-- TODO: GIF do demo ao vivo aqui (letras sendo reconhecidas + o "?" ao coçar a cabeça) -->

---

## Como o sistema funciona

```
  webcam  ──►  MediaPipe  ──►  normalização  ──►  classificador  ──►  "C" (0.98)
  (frame)      (landmarks)     (vetor 63)         (LogReg)             texto + confiança
                                                       │
                                                       ▼
                                              detecção de novidade
                                              (longe do treino? → "?")
```

O ponto central da arquitetura: **não classificamos pixels.** O MediaPipe — uma
rede pré-treinada pelo Google — converte cada frame nas coordenadas 3D das
articulações das mãos. O frame de 921.600 bytes vira um vetor de **63 números**
(21 pontos × 3 eixos) que descreve apenas a **geometria** do gesto, já livre de
fundo, iluminação e aparência do sinalizante.

Isso é *transfer learning*, e é o que torna o projeto viável: bastam ~100–200
amostras por sinal, e o modelo treina em segundos numa CPU.

O raciocínio completo — inclusive o que foi descartado e por quê — está em
[docs/DECISOES.md](docs/DECISOES.md) (16 ADRs).

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

| modelo | acurácia | desvio entre sessões |
| ------ | -------- | -------------------- |
| **LogisticRegression** | **99,7%** | 0,1% |
| RandomForest | 97,1% | 2,1% |

*(5 letras — A, B, C, L, O — 2.567 amostras, 3 sessões em luzes diferentes.)*

> ⚠️ **Ressalva honesta.** As três sessões são da **mesma pessoa e da mesma mão**.
> O número prova generalização entre **luz e posição**, não entre **pessoas**.
> Uma mão com proporções diferentes é território que o modelo nunca viu.

Duas observações que valem mais que o número: o modelo **mais simples**
generalizou melhor (o RandomForest degradou na sessão inédita), e por isso a rede
neural **não foi construída** — com 99,7% em 5 classes linearmente separáveis não
há espaço para ela melhorar. A MLP entra quando o alfabeto crescer e os punhos
parecidos (M/N/S/T) tornarem o problema não-linear. Ver
[ADR-002](docs/DECISOES.md) e [ADR-015](docs/DECISOES.md).

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

# 5. o demo ao vivo
python scripts\reconhecer.py
```

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
| `tests/` | 58 testes (pytest). |
| `docs/` | `DECISOES.md` — 16 ADRs com o porquê de cada escolha. |
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

**O que não está:** as letras **com movimento** (H, J, K, X, Y, Z) exigem um
modelo temporal (LSTM) e ficam para a fase dinâmica — estão declaradas como
`dinamico` em [`sinais.yaml`](libras/registry/sinais.yaml) e o coletor as pula
automaticamente. Sinais de duas mãos e a camada de LLM (que junta as letras em
palavras e corrige a gramática) também são fases seguintes.

Não existe modelo de Libras pronto para usar: os modelos públicos são de **ASL**,
que é outra língua. O MediaPipe entrega landmarks, não significado — o
significado vem dos dados que você rotula.

---

## Stack

**Visão computacional:** OpenCV (captura) + MediaPipe (landmarks).
**ML:** scikit-learn (baseline e modelo atual) + PyTorch (previsto para a fase
não-linear).
**Qualidade:** pytest + ruff.
**Previsto:** FastAPI + Uvicorn (WebSocket a 30 FPS) e React + Vite + Tailwind.
