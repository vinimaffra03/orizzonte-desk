# Orizzonte Desk — checklist de release

Esta checklist produz uma versão verificável e **não envia ordens mainnet**.

## 1. Congelar a pesquisa

- Sincronizar e auditar o histórico Binance com papel `development`.
- Congelar configuração, seed, período e commit antes de obter o holdout.
- Sincronizar a janela Hyperliquid com papel `external_holdout`.
- Confirmar que nenhum comando de treino aceita o holdout.

## 2. Produzir o candidato

- Executar o walk-forward ancorado e todos os cenários de stress.
- Treinar o candidato final com a receita congelada.
- Confirmar que o gate longo tem escopo `training_protocol` e hash dos folds.
- Avaliar uma única vez no holdout Hyperliquid usando `--model-id`.
- Confirmar que o gate do holdout tem escopo `candidate`.
- Gerar o gate combinado e promover somente o hash avaliado, no mesmo config/code/commit.
- Se qualquer gate falhar, manter testnet/mainnet bloqueados e publicar o relatório negativo.

## 3. Validar operação

- Criar uma API wallet exclusiva para o processo.
- Executar preflight e smoke somente no testnet, com saldo mock.
- Verificar entrada, fill parcial, SL/TP reduce-only, restart, reconciliação e flatten.
- Confirmar que não restaram ordens, posições ou proteções órfãs.

## 4. Construir e aprovar a release

Com o worktree limpo e todos os artefatos locais presentes:

```powershell
uv run orizzonte release build
uv run orizzonte release verify <release-id>
uv run orizzonte release approve <release-id>
```

A confirmação deve ser exatamente `APPROVE RELEASE <release-id>`. A aprovação vincula commit,
configuração, modelo e gate por SHA-256. Qualquer alteração posterior invalida o preflight.

## 5. Aceite

- CI verde em Linux e Windows.
- Nenhum segredo, dataset, modelo ou relatório versionado.
- `orizzonte doctor`, `release verify` e `testnet preflight` aprovados.
- Conta sem ordens/posições manuais.
- Mainnet permanece desarmada. Uma sessão real exige uma decisão operacional futura e separada.
