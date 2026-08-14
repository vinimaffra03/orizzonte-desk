# Orizzonte Desk

Orizzonte Desk é um ambiente quantitativo controlado por terminal para pesquisa, backtest, paper trading e execução de perpétuos **BTC, ETH, SOL e XRP** na Hyperliquid.

O núcleo combina setups determinísticos multi-timeframe com um meta-modelo LightGBM calibrado. O modelo apenas filtra sinais; limites de risco, sizing, execução e kill switches são determinísticos.

> **Risco:** a meta de 1% líquido ao dia é um benchmark de pesquisa, não garantia. O sistema não força operações, não usa martingale e pode perder capital. Alavancagem de 10× amplia perdas e risco de liquidação. Use testnet e revise os relatórios antes de considerar mainnet.

## Segurança operacional

- Universo fixo: BTC, ETH, SOL e XRP.
- Margem isolada e alavancagem configurada em 10×.
- Risco máximo padrão de 1% do orçamento por trade e 2% agregado.
- Até duas posições simultâneas, com bloqueio de altcoins correlacionadas.
- Profit lock em +1% UTC, stop diário em −4% e kill switch em −25% desde o high-water mark.
- SL/TP reduce-only instalados na exchange; falha de proteção fecha a posição.
- API wallets exclusivas e separadas para testnet/mainnet, protegidas por DPAPI. A chave
  principal não deve ser usada e fingerprints rotacionadas não podem ser reutilizadas.
- O adaptador não implementa saque ou transferência.
- Mainnet exige modelo promovido, dois gates aprovados, release aprovada, certificado
  testnet e capability DPAPI de uso único com TTL de 15 minutos e teto de 500 USDC.

## Instalação no disco D:

Requisitos: Windows, PowerShell, Git e [`uv`](https://docs.astral.sh/uv/).

```powershell
Set-Location 'D:\orizzonte desk'
. .\scripts\bootstrap.ps1
```

O script direciona cache, runtime Python e temporários para dentro da pasta do projeto no disco D:. Builds Docker devem ser executados apenas no GitHub Actions enquanto o Docker Desktop estiver armazenando imagens no C:.

## Fluxo de pesquisa

Para uma demonstração inteiramente local:

```powershell
uv run orizzonte data sync --source synthetic --hours 16000
uv run orizzonte data validate
uv run orizzonte research train
uv run orizzonte backtest run
uv run orizzonte report open
```

Para a pesquisa exigida antes de live:

```powershell
# Histórico longo multi-venue
uv run orizzonte data sync --source binance --start 2021-01-01
uv run orizzonte backtest run

# Validação recente e externa na Hyperliquid
uv run orizzonte data sync --source hyperliquid --environment mainnet
uv run orizzonte backtest run

# Combine explicitamente os dois gates
uv run orizzonte backtest compare
```

Um gate aprovado exige Sharpe ≥ 1,0, profit factor ≥ 1,15, drawdown ≤ 25%, stress positivo, três ativos positivos e probabilidade Monte Carlo de perda de 50% inferior a 1%.

## Modelo e promoção

```powershell
uv run orizzonte research train
uv run orizzonte research diagnose --dataset-id <dataset-development>
uv run orizzonte research regimes --dataset-id <dataset-development>
uv run orizzonte research evaluate
uv run orizzonte research promote model-AAAAMMDDTHHMMSSZ --gate 'D:\orizzonte desk\reports\<run>\gate.json'
```

Treinar cria um candidato e uma `DecisionPolicy` content-addressed. O threshold é escolhido
por split temporal nested, purge de 24 horas, custos 2× e block bootstrap; sem limite inferior
positivo, o modelo fica `no-trade` e não pode ser promovido. Somente `promote` altera o modelo
operacional, e apenas com gate aprovado. Baselines sem ML são diagnósticos e nunca geram gate.

O estudo de regimes é sempre *challenger*. Ele compara pooled, mapeamento estático e seletor
semanal em walk-forward event-driven, mas não pode substituir a v2 no mesmo ciclo nem olhar o
holdout Hyperliquid.

## Paper, testnet e mainnet

O daemon é o único escritor do estado operacional. CLI e TUI usam exclusivamente a API
HTTP/WebSocket em `127.0.0.1` para consultar estado e executar controles; se o daemon estiver
offline, os controles falham fechados. Ele executa o heartbeat, renova o dead man's switch e
avalia cada novo candle:

```powershell
uv run orizzonte daemon
```

Em outro terminal:

```powershell
uv run orizzonte tui
uv run orizzonte paper start --budget-usdc 10000
```

Configure uma **API wallet nova e exclusiva** autorizada na Hyperliquid. Informe o endereço da conta principal, não o endereço da API wallet:

```powershell
uv run orizzonte secret generate --environment testnet
uv run orizzonte secret verify --environment testnet
uv run orizzonte secret status --environment testnet
```

Antes de considerar uma liberação, construa e verifique o pacote imutável, faça o preflight e
execute manualmente o smoke test no **testnet**:

```powershell
uv run orizzonte release build
uv run orizzonte release verify release-<id>
uv run orizzonte release approve release-<id> # exige APPROVE RELEASE release-<id>
uv run orizzonte testnet preflight
uv run orizzonte testnet smoke --budget-usdc 25 # exige TESTNET SMOKE 25.00
uv run orizzonte testnet reconcile
uv run orizzonte testnet certificate
```

O fluxo quantitativo não atribui ao modelo final métricas que vieram dos modelos de cada fold.
O histórico longo gera um gate de **protocolo walk-forward**; o holdout Hyperliquid recente exige
`backtest run <dataset> --model-id <candidato>`. `backtest compare` só aprova quando protocolo e
candidato compartilham configuração, código e commit, e vincula ambos os datasets ao hash exato
do modelo que poderá ser promovido.

O smoke test deve validar entrada, fill, proteções, restart/reconciliação e flatten usando
saldo de testnet. Ele é deliberadamente manual e nunca troca silenciosamente para mainnet.
Armar testnet exige digitar exatamente `ORIZZONTE LIVE <AMBIENTE> <ORÇAMENTO>`; por exemplo:

```powershell
uv run orizzonte live arm --environment testnet --budget-usdc 1000
uv run orizzonte live start
uv run orizzonte live pause
uv run orizzonte live flatten   # exige FLATTEN
uv run orizzonte live disarm
```

O agente recusa armar se encontrar posições ou ordens preexistentes na conta. Mainnet continua
travada até uma autorização separada, vinculada a release, commit, configuração, modelo, gates,
certificado, conta, API wallet e orçamento:

```powershell
uv run orizzonte secret generate --environment mainnet
uv run orizzonte mainnet status
uv run orizzonte mainnet authorize --budget-usdc 500
# exige: AUTHORIZE MAINNET <release> <conta-lowercase> 500.00
uv run orizzonte live arm --environment mainnet --budget-usdc 500
```

Autorizar não arma nem envia ordem. `live arm` consome a capability atomicamente. Restart ou
pause exige uma nova autorização para entradas/resume, enquanto a gestão de posições e proteções
continua em `PAUSED`/`LOCKED`. Nesta entrega nenhum segredo/capability real é criado e nenhuma
ordem mainnet faz parte do aceite.

Para operação local no Windows:

```powershell
uv run orizzonte ops install       # exige INSTALL ORIZZONTE TASKS
uv run orizzonte ops status
uv run orizzonte ops backup
uv run orizzonte ops restore-dry-run <backup-id>
uv run orizzonte ops uninstall     # exige REMOVE ORIZZONTE TASKS
```

Os logs rotacionam no disco D:, o watchdog aplica lock fail-closed e os 30 backups mais recentes
são mantidos localmente. A instalação do daemon nunca autoriza retomar entradas após restart.

Siga a [checklist completa de release](docs/RELEASE_CHECKLIST.md) e o
[protocolo v2](docs/V2_PROTOCOL.md) para congelamento dos dados, gates, estudo challenger,
testnet e verificação dos hashes.

## CLI

```text
orizzonte init | doctor | daemon | tui
orizzonte data sync | validate | status
orizzonte research train | evaluate | diagnose | regimes | promote
orizzonte backtest run | stress | compare
orizzonte report latest | open | export
orizzonte paper start | pause | stop
orizzonte live arm | start | pause | resume | flatten | disarm
orizzonte secret generate | verify | rotate | status
orizzonte testnet preflight | smoke | reconcile | certificate
orizzonte release build | verify | approve
orizzonte mainnet authorize | status | revoke
orizzonte ops install | status | backup | restore-dry-run | uninstall
orizzonte status | positions | orders | risk | logs
```

## Artefatos locais

- `data/`: Parquet/Zstd e manifestos SHA-256.
- `models/`: candidatos e modelo promovido.
- `reports/`: HTML, CSV, JSON e gates.
- `state/`: SQLite em WAL.
- `.secrets/`: cofre DPAPI.
- `logs/`: logs estruturados.

Dados, modelos, relatórios, logs e segredos são ignorados pelo Git. O repositório público contém somente código, configurações de exemplo, testes e documentação.

## Desenvolvimento

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv run pip-audit
```

O build Docker é verificado no GitHub Actions. O container é preparado somente para
pesquisa/paper e implantação futura; **live em Linux permanece bloqueado**. O cofre DPAPI é
específico do Windows e, antes de qualquer live em Linux, será necessário conectar e auditar um
gerenciador de segredos externo, sem copiar uma chave privada em texto puro para a imagem.

## Limitações conhecidas

- O snapshot de candles da Hyperliquid contém no máximo 5.000 candles; por isso a pesquisa mantém um teste longo multi-venue e outro recente na venue.
- Funding histórico da validação Hyperliquid deve ser coletado continuamente para elevar a fidelidade; o backtest aplica custos conservadores enquanto isso.
- Nenhum resultado histórico elimina risco de regime, liquidez, slippage, falha de rede, ADL ou liquidação.
- TradingView, Pine Script, webhooks e conectores TradingView ficam fora da v1; sinais e execução usam somente dados e APIs oficiais integrados ao daemon.

Licenciado sob Apache-2.0.
