# OmniVoice — Instalacao no Windows

Guia rapido para instalar e rodar o OmniVoice no Windows com placa NVIDIA.

## Pre-requisitos

- **Windows 10 ou 11** (64 bits)
- **Placa NVIDIA** com driver atualizado (recomendado RTX 20xx, 30xx, 40xx)
- **Espaco em disco:** ~10 GB (o modelo baixa na primeira execucao)
- **Conexao com internet** (so na primeira execucao, para baixar o modelo)

## Passo 1 — Instalar Git

Baixe e instale: https://git-scm.com/download/win

Aceite as opcoes padrao no instalador.

## Passo 2 — Instalar uv (gerenciador Python)

Abra o **PowerShell** (botao direito no menu Iniciar, "Windows PowerShell") e cole:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Feche e reabra o PowerShell depois que instalar.

## Passo 3 — Clonar o repositorio

No PowerShell, escolha onde quer instalar (ex: `C:\OmniVoice`) e rode:

```powershell
cd C:\
git clone https://github.com/godoiads/OmniVoice.git
cd OmniVoice
```

## Passo 4 — Instalar dependencias

Ainda no PowerShell, dentro da pasta `OmniVoice`:

```powershell
uv sync
```

Isso vai baixar o PyTorch com CUDA e todas as dependencias. **Demora uns 5-15 minutos** dependendo da internet.

## Passo 5 — Rodar pela primeira vez

De um duplo clique no arquivo **`OmniVoice.bat`** dentro da pasta.

- Na primeira execucao, o modelo (~3 GB) sera baixado automaticamente do HuggingFace. **Demora 5-20 minutos**.
- Quando estiver pronto, o navegador abre automaticamente em http://localhost:8001
- Para parar: feche a janela preta do `OmniVoice.bat`.

## Como usar

A interface tem 3 abas:

1. **Clonagem de voz** — solta um audio de referencia (3-10 segundos) e escreve o texto que voce quer falar com aquela voz.
2. **Design de voz** — escolhe atributos (genero, idade, sotaque) sem precisar de audio de referencia.
3. **Voz automatica** — voz aleatoria gerada pelo modelo.

Voce pode **salvar vozes clonadas** com um nome (ficam na pasta `voices/`) e reutilizar depois.

## Problemas comuns

**"python.exe nao foi encontrado"**
- O `uv sync` nao terminou ou deu erro. Volte ao Passo 4 e rode de novo.

**"CUDA out of memory"**
- Sua GPU tem pouca VRAM. Tente fechar outros programas. O modelo precisa de ~4 GB de VRAM livre.

**Navegador nao abre sozinho**
- Espere o terminal mostrar "Running on local URL: http://0.0.0.0:8001" e abra manualmente.

**Quer rodar sem GPU (so CPU)**
- Funciona, mas e MUITO lento (40-100x mais lento). Nao recomendado.

## Atualizacao futura

Para puxar atualizacoes do fork:

```powershell
cd C:\OmniVoice
git pull
uv sync
```

## Contato

Qualquer problema, fala com o Gustavo.
