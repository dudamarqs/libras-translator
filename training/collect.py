"""Coletor interativo do dataset da datilologia.

    .venv\\Scripts\\python.exe training\\collect.py
    .venv\\Scripts\\python.exe training\\collect.py --letras "A B C L O" --segundos 6

Fluxo: para cada letra, voce se posiciona, um contador 3-2-1 dispara, e o
coletor grava por alguns segundos todos os frames em que UMA mao e detectada.
Ao final, tudo vira UMA sessao em datasets/raw/<session_id>/.

--------------------------------------------------------------------------
COMO COLETAR BEM (leia -- a qualidade do dataset e o teto do projeto)
--------------------------------------------------------------------------

DURANTE a gravacao de cada letra, MOVA a mao:
  - para os lados e para cima/baixo (posicao no quadro),
  - para perto e para longe da camera (escala),
  - incline levemente a mao (rotacao).

As duas primeiras a `features.py` ja neutraliza -- mas variar mesmo assim nao
custa nada e da robustez. A terceira, a INCLINACAO, e a que importa de verdade:
decidimos NAO normalizar a rotacao (ADR-012), porque a orientacao da palma e
gramatica em Libras. Entao o modelo so vai tolerar inclinacao se ele a VIR aqui.
Se voce gravar a letra sempre no mesmo angulo exato, ela so sera reconhecida
naquele angulo.

FACA VARIAS SESSOES, idealmente em dias/luzes diferentes. Frames de uma mesma
sessao sao parecidos demais entre si; e a variedade ENTRE sessoes que ensina o
modelo a generalizar. O split treino/teste e por sessao justamente por isso.

Teclas:  ESPACO = comecar a proxima letra   |   s = pular letra
         r = regravar a ultima letra         |   q = salvar e sair
"""

from __future__ import annotations

import argparse
import subprocess
import time

import cv2
import numpy as np

from libras.data.dataset import Sessao, novo_session_id, salvar_sessao
from libras.registry import registro
from libras.vision.camera import Camera, CameraConfig, CameraError
from libras.vision.hands import DetectorMaos
from libras.vision.overlay import AMARELO, VERDE, VERMELHO, desenhar_maos, painel, texto

COLETOR_VERSAO = 1
MIN_AMOSTRAS_ALERTA = 60


def _commit_atual() -> str:
    """Hash do commit de codigo -- provenance. Best-effort."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "desconhecido"


def _esperar_inicio(cam: Camera, detector: DetectorMaos, letra: str) -> str:
    """Tela de preparacao. Retorna a acao escolhida: 'go' | 'skip' | 'quit'."""
    while True:
        frame = cam.ler()
        resultado = detector.detectar(frame)
        desenhar_maos(frame, resultado.maos)

        cor = VERDE if not resultado.vazio else AMARELO
        painel(
            frame,
            [
                f"PROXIMA LETRA:  {letra}",
                "faca a letra e ESPACO para gravar",
                "s = pular   r = regravar ultima   q = salvar e sair",
                "mao detectada" if not resultado.vazio else "MOSTRE A MAO",
            ],
            cor=cor,
        )
        cv2.imshow("Coleta", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord(" "):
            return "go"
        if tecla == ord("s"):
            return "skip"
        if tecla == ord("r"):
            return "redo"
        if tecla in (ord("q"), 27):
            return "quit"


def _contagem_regressiva(
    cam: Camera, detector: DetectorMaos, letra: str, segundos: int = 3
) -> None:
    fim = time.perf_counter() + segundos
    while (restante := fim - time.perf_counter()) > 0:
        frame = cam.ler()
        resultado = detector.detectar(frame)
        desenhar_maos(frame, resultado.maos)
        texto(
            frame,
            f"{letra}  ...  {int(restante) + 1}",
            (frame.shape[1] // 2 - 80, frame.shape[0] // 2),
            escala=2.0,
            cor=AMARELO,
            espessura=3,
        )
        cv2.imshow("Coleta", frame)
        cv2.waitKey(1)


def _gravar_letra(
    cam: Camera, detector: DetectorMaos, letra: str, segundos: float
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Grava por `segundos`, guardando (world, imagem, lado) de cada frame com mao."""
    coletadas: list[tuple[np.ndarray, np.ndarray, str]] = []
    fim = time.perf_counter() + segundos

    while (restante := fim - time.perf_counter()) > 0:
        frame = cam.ler()
        resultado = detector.detectar(frame)

        # Datilologia = 1 mao. Se aparecer mais de uma (ou nenhuma), pulamos o
        # frame -- so guardamos amostras limpas.
        if len(resultado.maos) == 1:
            mao = resultado.maos[0]
            coletadas.append((mao.world.copy(), mao.landmarks.copy(), mao.lado))

        desenhar_maos(frame, resultado.maos)
        painel(
            frame,
            [
                f"GRAVANDO  {letra}",
                "MOVA e INCLINE a mao!",
                f"{restante:3.1f}s   amostras: {len(coletadas)}",
            ],
            cor=VERMELHO,
        )
        cv2.imshow("Coleta", frame)
        cv2.waitKey(1)

    return coletadas


def _resumo(por_letra: dict[str, list]) -> None:
    print("\n" + "=" * 46)
    print("RESUMO DA SESSAO")
    print("=" * 46)
    print(f"{'letra':<8} {'amostras':>10}")
    print("-" * 46)
    for letra in sorted(por_letra):
        n = len(por_letra[letra])
        alerta = "  <- poucas!" if n < MIN_AMOSTRAS_ALERTA else ""
        print(f"{letra:<8} {n:>10}{alerta}")
    print("-" * 46)
    total = sum(len(v) for v in por_letra.values())
    print(f"{'TOTAL':<8} {total:>10}")

    # Alerta de DESBALANCEAMENTO. Um dataset onde 'A' tem 300 amostras e 'B' tem
    # 40 ensina o modelo a "chutar A quando na duvida". Balanceamento importa.
    contagens = [len(v) for v in por_letra.values() if v]
    if contagens and max(contagens) > 2 * min(contagens):
        print(
            "\n[ATENCAO] classes desbalanceadas (a maior tem mais que o dobro da\n"
            "menor). Considere regravar as letras com poucas amostras."
        )


def main() -> int:
    reg = registro()
    letras_estaticas = [s.nome for s in reg.estaticos()]

    parser = argparse.ArgumentParser(description="Coletor do dataset de datilologia")
    parser.add_argument(
        "--letras",
        type=str,
        default=" ".join(letras_estaticas),
        help="letras a coletar, separadas por espaco (padrao: todas as estaticas)",
    )
    parser.add_argument(
        "--segundos", type=float, default=6.0, help="segundos de gravacao por letra"
    )
    args = parser.parse_args()

    letras = args.letras.upper().split()
    # Validamos contra o registro ANTES de abrir a camera: erro de digitacao
    # ("--letras 'A B 8'") falha agora, nao depois de voce ja ter gravado 10
    # letras e perder a sessao.
    for letra in letras:
        sinal = reg.get(letra)  # levanta KeyError com mensagem clara
        if not sinal.estatico:
            print(f"[aviso] '{letra}' e dinamica (tem movimento) -- pulei. Fica para a fase LSTM.")
    letras = [c for c in letras if reg.get(c).estatico]

    print(f"\nVou coletar {len(letras)} letras: {' '.join(letras)}")
    print(f"{args.segundos:.0f}s por letra. Uma janela vai abrir.\n")

    config = CameraConfig()
    por_letra: dict[str, list] = {}

    try:
        with (
            Camera(config) as cam,
            DetectorMaos(max_maos=1, entrada_espelhada=config.espelhar) as detector,
        ):
            largura, altura = cam.resolucao_efetiva
            i = 0
            ultima: str | None = None
            while i < len(letras):
                letra = letras[i]
                acao = _esperar_inicio(cam, detector, letra)

                if acao == "quit":
                    break
                if acao == "skip":
                    i += 1
                    continue
                if acao == "redo" and ultima is not None:
                    por_letra.pop(ultima, None)
                    i = letras.index(ultima)
                    continue

                _contagem_regressiva(cam, detector, letra)
                coletadas = _gravar_letra(cam, detector, letra, args.segundos)
                por_letra.setdefault(letra, []).extend(coletadas)
                ultima = letra
                print(f"  {letra}: +{len(coletadas)} amostras (total {len(por_letra[letra])})")
                i += 1

    except CameraError as exc:
        print(f"\nERRO DE CAMERA: {exc}")
        return 1
    except FileNotFoundError as exc:
        print(f"\n{exc}")
        return 1
    finally:
        cv2.destroyAllWindows()

    total = sum(len(v) for v in por_letra.values())
    if total == 0:
        print("\nNenhuma amostra coletada. Nada foi salvo.")
        return 1

    # Achata o dict em arrays paralelos.
    worlds, imagens, lados, rotulos = [], [], [], []
    for letra, amostras in por_letra.items():
        for world, imagem, lado in amostras:
            worlds.append(world)
            imagens.append(imagem)
            lados.append(lado)
            rotulos.append(letra)

    sessao = Sessao(
        session_id=novo_session_id(),
        world=np.array(worlds, dtype=np.float32),
        imagem=np.array(imagens, dtype=np.float32),
        lado=np.array(lados),
        rotulo=np.array(rotulos),
    )
    destino = salvar_sessao(
        sessao,
        meta_extra={
            "coletor_versao": COLETOR_VERSAO,
            "commit": _commit_atual(),
            "camera": {"largura": largura, "altura": altura, "backend": "MSMF"},
            "segundos_por_letra": args.segundos,
        },
    )

    _resumo(por_letra)
    print(f"\nSessao salva em: {destino}")
    print("Rode o coletor de novo, outro dia, para acumular mais sessoes.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
