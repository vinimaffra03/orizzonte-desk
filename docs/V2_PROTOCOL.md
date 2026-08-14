# Protocolo Orizzonte Desk v2

Este documento define o ciclo de pesquisa e as travas operacionais da v2. Ele é um
contrato de reprodutibilidade, não uma promessa de retorno. A meta de 1% ao dia é
apenas uma métrica informativa.

## Pesquisa oficial

1. Congele commit, `config/settings.toml`, seed, dataset Binance de desenvolvimento e
   capital inicial de 10.000 USDC.
2. Execute `research diagnose` e `research regimes`. O estudo de regimes é sempre um
   *challenger* e não pode substituir a estratégia v2 nesse ciclo.
3. Execute o walk-forward v2 duas vezes com os mesmos inputs. `run_id`, hashes,
   métricas, equity e trades devem ser idênticos.
4. Interrompa o ciclo se o gate longo reprovar. Não baixe o holdout.
5. Se aprovado, treine e congele um único candidato. Só então baixe a janela recente
   da Hyperliquid com papel `external_holdout` e avalie-a uma única vez com o ID exato
   do candidato.
6. Combine os dois gates. Divergência de modelo, dataset, configuração, código,
   commit ou protocolo invalida a execução.

O threshold operacional é uma `DecisionPolicy` content-addressed. Em cada fold ela é
selecionada somente no split interno temporal, após purge de 24 horas, por quantis de
50% a 95%, custos duplicados e block bootstrap determinístico. São necessários 30
trades internos para avaliar um quantil e o limite inferior de 5% da expectativa
líquida em R precisa ser positivo. Sem quantil elegível, o fold fica sem operar.
Modelos sem `DecisionPolicy` não podem ser promovidos nem usados em testnet/live.

Baselines sem ML e o estudo de regimes geram diagnóstico, mas nunca gate ou promoção.

## Estudo de regimes

A decisão semanal ocorre às segundas-feiras, 00:05 UTC, usando somente candles
completamente fechados e evidência encerrada pelo menos 24 horas antes. A decisão é
persistida até a semana seguinte; restart apenas restaura o registro existente.

São comparados pooled, breakout, pullback, mapeamento estático, seletor semanal e flat,
sem ML e com a mesma política ML nested. O braço semanal só é superior quando cumpre
simultaneamente todos os critérios definidos no plano v2. Se o estático ficar a menos
de 0,10 de Sharpe, ele é preferido pela simplicidade. Um resultado favorável abre uma
pesquisa v3; nunca muda a v2 já exposta ao holdout.

## Testnet e mainnet

Os estados de paper, testnet e mainnet são isolados por ambiente e conta. Os cofres
DPAPI de testnet e mainnet são distintos. Uma release aprovada ainda não habilita
mainnet.

O certificado testnet é content-addressed e vincula release, modelo, gates, conta,
API wallet testnet e evidências do smoke/caos. A autorização mainnet é outra capability:
DPAPI, uso único, validade de 15 minutos, teto de 500 USDC e vínculos exatos com a
release, certificado, commit, configuração, modelo, gates, conta e API wallet mainnet.

Reiniciar em mainnet sempre resulta em `PAUSED`. Posições continuam sendo reconciliadas
e protegidas em `PAUSED` ou `LOCKED`, mas nenhuma nova entrada é permitida sem nova
autorização. O dead man's switch só é renovado enquanto a conta está flat ou há entrada
pendente; após confirmar posição e duas proteções reduce-only ele é removido para não
cancelar SL/TP durante uma queda do daemon.

Nesta entrega não são criados segredos reais, capabilities reais ou ordens. Mainnet
deve permanecer impossível até o operador concluir deliberadamente todas as etapas.

## Aceite local

```powershell
uv sync --frozen
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv run pip-audit
uv run orizzonte doctor
```

Todos os artefatos volumosos, temporários, bancos, modelos, relatórios e backups ficam
em `D:\orizzonte desk`. A execução deve parar antes de o espaço livre cair abaixo de
20 GB. Docker é validado somente pelo GitHub Actions enquanto o armazenamento global
do Docker Desktop permanecer no disco C:.
