# Orizzonte Desk — checklist de release

Esta checklist produz uma versão verificável e **não envia ordens mainnet**.

## 1. Congelar a pesquisa

- Sincronizar e auditar o histórico Binance com papel `development`.
- Congelar configuração, seed, período e commit antes de obter o holdout.
- Executar `research diagnose` e o challenger `research regimes` somente no development.
- Confirmar threshold nested temporal, purge de 24h, bootstrap determinístico e
  `DecisionPolicy` content-addressed. Fold sem LCB positivo deve ficar sem operar.
- Rodar a v2 oficial duas vezes e comparar run ID, trades, equity, métricas e hashes.
- Sincronizar a janela Hyperliquid como `external_holdout` somente se o gate longo passar.
- Confirmar que nenhum comando de treino aceita o holdout.

## 2. Produzir o candidato

- Executar o walk-forward ancorado e todos os cenários de stress.
- Treinar o candidato final com a receita congelada.
- Confirmar que o gate longo tem escopo `training_protocol` e hash dos folds.
- Avaliar uma única vez no holdout Hyperliquid usando `--model-id`.
- Confirmar que o gate do holdout tem escopo `candidate`.
- Gerar o gate combinado e promover somente o hash avaliado, no mesmo config/code/commit.
- Se qualquer gate falhar, manter testnet/mainnet bloqueados e publicar o relatório negativo.

## 3. Construir e aprovar a release

Com o worktree limpo e todos os artefatos locais presentes:

```powershell
uv run orizzonte release build
uv run orizzonte release verify <release-id>
uv run orizzonte release approve <release-id>
```

A confirmação deve ser exatamente `APPROVE RELEASE <release-id>`. A aprovação vincula commit,
configuração, modelo e gate por SHA-256. Qualquer alteração posterior invalida o preflight.

## 4. Certificar testnet

- Gerar uma API wallet testnet exclusiva com `secret generate --environment testnet`.
- Executar preflight, smoke e caos somente no testnet, com saldo fornecido pelo operador.
- Verificar entrada, fill parcial, duas proteções reduce-only, restart, reconciliação,
  duplicatas, timeout após aceite, clock drift, stale data, falha de proteção, dead man's
  switch e flatten.
- Confirmar conta vazia ao final e `TestnetCertificate` válido, vinculado à release/modelo/gates.

## 5. Aceite

- CI verde em Linux e Windows.
- Nenhum segredo, dataset, modelo ou relatório versionado.
- `orizzonte doctor`, `release verify` e `testnet preflight` aprovados.
- Conta sem ordens/posições manuais.
- Mainnet permanece desarmada e sem capability. Autorização futura exige cofre mainnet
  separado, certificado exato, confirmação completa, TTL de 15 minutos e budget ≤ 500 USDC.
