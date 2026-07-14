# Tradutor de Libras em Tempo Real

Reconhecimento de sinais de Libras pela webcam, traduzidos para texto em tempo
real — com a confiança da predição exibida junto.

> **Status:** em construção. Etapa 2/19 (configuração do ambiente) concluída.

---

## Como o sistema funciona

```
  webcam  ──►  MediaPipe  ──►  normalização  ──►  classificador  ──►  "OI" (0.94)
  (frame)      (landmarks)     (vetor 126)        (MLP / LSTM)         texto + confiança
```

O ponto central da arquitetura: **não classificamos pixels.** O MediaPipe — uma
rede pré-treinada pelo Google — converte cada frame nas coordenadas 3D das
articulações das mãos. O frame de 150.528 números vira um vetor de ~126 números
que descreve apenas a **geometria** do gesto, já livre de fundo, iluminação e
aparência do sinalizante.

Isso é *transfer learning*, e é o que torna o projeto viável: bastam ~100–200
amostras por sinal, e o modelo treina em segundos numa CPU.

O raciocínio completo — inclusive o que foi descartado e por quê — está em
[docs/DECISOES.md](docs/DECISOES.md).

---

## Estrutura

| Pasta        | Função |
| ------------ | ------ |
| `libras/`    | **Pacote compartilhado.** Extração e normalização de landmarks, datasets, arquiteturas de modelo e o catálogo de sinais. Importado *tanto* pelo treino *quanto* pelo backend — é isso que impede o `training/serving skew` (ADR-004). |
| `training/`  | Scripts executáveis do ciclo de ML: `collect.py`, `preprocess.py`, `train.py`, `evaluate.py`. |
| `backend/`   | API FastAPI: rotas HTTP, WebSocket de inferência, schemas Pydantic. |
| `frontend/`  | Interface React + Vite + Tailwind. |
| `notebooks/` | Experimentação exploratória. Nada aqui vira produção — o que der certo migra para `libras/`. |
| `datasets/`  | `raw/` (o que a webcam capturou) e `processed/` (features prontas para treino). Fora do Git. |
| `models/`    | Checkpoints treinados + `metadata.json` (classes, métricas, versão). Fora do Git. |
| `tests/`     | Testes automatizados. |
| `scripts/`   | Utilidades de ambiente, como o `doctor.py`. |
| `docs/`      | ADRs e documentação. |

`datasets/` e `models/` ficam **fora do Git**: o Git guarda a história completa
de cada arquivo, então um dataset de 500 MB regravado 10 vezes vira um repo de
5 GB — para sempre, mesmo depois de você deletar o arquivo.

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

# 5. verificar que tudo realmente funciona
python scripts\doctor.py
```

O `doctor.py` valida o que `pip install` **não** valida: se numpy/OpenCV/MediaPipe
trocam arrays sem estourar a ABI em C, se a webcam entrega frame de verdade, e
se o pacote `libras` está importável.

---

## Stack

**Visão computacional:** OpenCV (captura) + MediaPipe (landmarks).
**ML:** PyTorch (modelo) + scikit-learn (baseline — sempre antes da rede neural).
**Backend:** FastAPI + Uvicorn (WebSocket para inferência a 30 FPS).
**Frontend:** React + Vite + Tailwind.
**Qualidade:** pytest + ruff.
