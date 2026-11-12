---
primary_layer: layer3
related_layers: [layer2]
page_type: concept
---

# OptimizationStrategy

`saealib` は、高コストな真の評価へ送る候補の選択を `OptimizationStrategy` に委譲します。 Strategyは、候補生成、予測、評価計画、Feedback、Population更新を一つのグラフとして記述します。

## OptimizationStrategyの役割

Strategyは、候補生成、予測、評価計画、Feedback、Population更新を一つの宣言フローとして記述します。 正規の書き方は、Stageが `(slot, role)` でコンポーネントを参照する `Pipeline` 宣言を `TemplateStrategy` に渡す形です。 実行前にOptimizerはこの宣言を `ComponentGraph` へ下ろしてCompilerに渡し、`ExecutablePlan` を作成します。

`build_graph(provider) -> ComponentGraph` は宣言を下ろす境界として残りますが、独自Strategyでの上書きは非推奨です。 上書きすると `DeprecationWarning` が出力され、代替は `Pipeline` 宣言を渡した `TemplateStrategy` です。 フローをこの方法で検査できないStrategyは、引き続き `requires_surrogate: bool` を手で宣言します。

`requires_surrogate` は、この戦略が`SurrogateManager`に対して宣言する要件を持つ通常の属性です。 `OptimizationStrategy` が `False` を既定値として定義し、`TemplateStrategy` は宣言に判断を委ねるため `None` のままにします。 `DirectStrategy` と `SteadyStateStrategy` は `False` を、`IndividualBasedStrategy` と `PreSelectionStrategy` は `True` を宣言し、`GenerationBasedStrategy` は何も宣言しないため、surrogateのみの世代を持つかどうかによらず属性は `None` です。 通常の属性なので、クラス属性、Mixinから継承した属性、インスタンスへの代入、pickleやspecから復元した値のいずれもStrategyの宣言として扱われます。

この属性は実効要件ではありません。 標準のloweringを使う `TemplateStrategy` では、構成診断が宣言フローから要件を導出します。 宣言内のどのStage（ネストしたRepeat、Loop、Branch領域の内部も含む）が `ComponentRef` でsurrogate managerスロットを参照するか、surrogate managerを直接保持している場合に `True` になり、その値が `missing_surrogate_manager` と `missing_acquisition` エラーを決めます。 したがって `strategy.requires_surrogate` を真偽値として読むコードはそれを実効要件として扱えません。 構成を確認するには `Optimizer.validate()` か `Optimizer.contract_diagnostics()` の `missing_surrogate_manager` を使います。 導出された要件と食い違う宣言値は `contradicting_surrogate_requirement` WARNING診断として報告され、実行は継続します。 `build_graph` を上書きした戦略は宣言した属性をそのまま使います。

## 組み込み省略形と明示的な方針コンポーネント

組み込みコンストラクタは、宣言と既定の方針コンポーネントを組み合わせた省略形です。 `IndividualBasedStrategy(evaluation_ratio)` は `RatioEvaluation(evaluation_ratio, sanitize_nonfinite=True)` を、`PreSelectionStrategy(n_candidates, n_select)` は `TopKEvaluation(min(n_select, n_candidates), sanitize_nonfinite=True)` を既定plannerとして束ね、それぞれ既定のfeedback builderも供給します。 両形式は固定シードで同じ計画と同じ結果になります。

`set_evaluation_planner` や `set_feedback_builder` などのsetterで明示設定したコンポーネントは省略形の既定より優先され、provenanceに明示設定として記録されます。 候補数（`n_candidates`、`n_offspring`）と内側世代数（`gen_ctrl`）は宣言の形そのものを決めるため、方針コンポーネントへは移らずコンストラクタ引数に留まります。 コンストラクタ引数はspecの往復とpickle復元でも使われます。

省略形パラメータのin-place変更（例：`strategy.evaluation_ratio = 0.5`）は、次の確定点、つまり戦略が次に計画用に宣言を解決する時点で効きます。 明示設定されたコンポーネントには決して影響せず、そのコンポーネントは自身の設定を保ちます。

## 組み込みStrategy

| クラス | パラメータ | 方式 |
|---|---|---|
| `DirectStrategy` | なし | サロゲートを使わず、生成した候補を全件真に評価する |
| `SteadyStateStrategy` | なし | 1ステップにつき1候補を生成し、非同期補充に対応する |
| `IndividualBasedStrategy` | `evaluation_ratio: float = 0.1` | 全候補をサロゲートでスコアリングし、上位`evaluation_ratio`の割合だけ真に評価する |
| `PreSelectionStrategy` | `n_candidates: int, n_select: int`（共に必須） | `n_candidates`件を生成してスコアリングし、上位`n_select`件だけ真に評価する |
| `GenerationBasedStrategy` | `gen_ctrl: int`（必須） | `gen_ctrl`世代分をサロゲートのみで進め、その後1世代だけ真に評価する |

`IndividualBasedStrategy`は個体の割合、`PreSelectionStrategy`は個体の件数で選抜する点が異なります。 `GenerationBasedStrategy`は個体単位ではなく世代単位でサロゲートと真の評価を切り替えます。 `DirectStrategy`はサロゲートを一切使わない比較対象です。

定常状態実行では、`DirectStrategy`と`AsyncEvaluationScheduler`を組み合わせます。スケジューラは利用可能なワーカースロットを埋め、ブロックせずにポーリングし、完了した更新をライフサイクル順にコミットします。

### 各Strategyのパイプライン構成

| クラス | パイプライン |
|---|---|
| `DirectStrategy` | 世代数のカウント → 要求 → 真の評価 → アーカイブ更新 → 反映 |
| `IndividualBasedStrategy` | CountGeneration → Ask → SurrogatePredictStage → AcquisitionStage → SortByScore → TrueEvaluation（比率指定）→ ArchiveUpdate → Tell |
| `PreSelectionStrategy` | CountGeneration → Ask（n_candidates）→ SurrogatePredictStage → AcquisitionStage → TopKSelection（k=n_select）→ TrueEvaluation → ArchiveUpdate → Tell |
| `GenerationBasedStrategy` | SurrogateOnlyLoop(gen_ctrl回) → CountGeneration → Ask → TrueEvaluation → ArchiveUpdate → Tell |

各Strategyの構成要素には、`EvaluationPlan`、非同期評価の送信と収集、Feedback、Tellが含まれます。Stageはその一部を担う互換性用の実行単位であり、構造化されたパイプライン全体と同じ実行契約ではありません。詳細は[Stage](../observation_and_state/stage.md)と[Framework](../../framework/index.md)を参照してください。

## どのStrategyを選ぶか

評価コストが極めて高い問題では、サロゲートで大半の候補を足切りする`IndividualBasedStrategy`や`PreSelectionStrategy`が有効です。 サロゲートの信頼度がまだ低い探索初期や、サロゲートの学習コスト自体を頻繁に払いたくない場合は、複数世代をまとめてサロゲートのみで進める`GenerationBasedStrategy`が適しています。 サロゲートの近似誤差そのものが許容できない、あるいは評価コストが十分低い問題では、`DirectStrategy`で真の評価だけに頼るのが妥当です。

## 実行時差し替えの挙動

`iterate()` / `run()` のステップ境界で `provider.algorithm` や `provider.surrogate_manager` などの構成を変更すると、実行環境が変化を検出してplanを再コンパイルし、次世代から反映します。 この手順はStage互換経路とgraph-native経路のどちらでも利用できます。 Component側から再コンパイルを要求するときは、plan nodeが `NodeResult.commands` で `RuntimeCommand` / `RequestRecompile` を返します。 要求はRuntime環境が `recompile()` を提供する場合に受理され、再コンパイルはステップ間で行われます。受理されないことも正常な結果です。

## 独自Strategyの実装方法

`(slot, role)` でコンポーネントを参照するStageを持つ `Pipeline` 宣言を `TemplateStrategy` に渡して、フローを宣言します。 コンポーネントは計画準備時にOptimizerのスロットから供給され、戦略は自身が所有するスロットの既定値を束ねられます。

```python
from saealib.pipeline import ComponentRef, Pipeline
from saealib.strategies import TemplateStrategy
from saealib.stages import AskStage, CountGenerationStage, EvaluationPlanStage


class CustomStrategy(TemplateStrategy):
    """Skeleton for a declared candidate-selection flow."""

    def __init__(self, n_offspring):
        self.n_offspring = n_offspring
        super().__init__(
            Pipeline(
                steps=[
                    CountGenerationStage(),
                    AskStage(ComponentRef("proposer"), n_offspring=n_offspring),
                    EvaluationPlanStage(ComponentRef("evaluation_planner")),
                ]
            )
        )
```

評価の送信、Feedback、TellのStageを構成に応じて追加します。 surrogate参照（`ComponentRef("predictor")`) はフローが必要な場合だけ宣言します。 surrogate要否は宣言から導出されます。

既存のStrategyの一部だけを調整する場合は、新しいStrategyを作るよりも [Pipeline.replace/find](../extension_guidelines.md) で互換性用パイプラインを変更する方が適しています。

## 関連コンポーネント

- [Stage](../observation_and_state/stage.md)：各Strategyが組み合わせるパイプラインステージ単体の契約
- [SurrogateManager](../surrogate_modeling/surrogate_manager.md)：`requires_surrogate=True`の戦略が使う予測機構
- [拡張のガイドライン](../extension_guidelines.md)：`Pipeline.replace`/`find`によるステージの並べ替え
- [コンポーネント概要](../index.md)：パイプライン全体の構成図

## 参照

- {py:class}`saealib.OptimizationStrategy`
- {py:class}`saealib.IndividualBasedStrategy`
- {py:class}`saealib.GenerationBasedStrategy`
- {py:class}`saealib.PreSelectionStrategy`
- {py:class}`saealib.DirectStrategy`
- {py:class}`saealib.AsyncEvaluationScheduler`
