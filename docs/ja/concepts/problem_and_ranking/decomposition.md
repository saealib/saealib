---
primary_layer: layer2
related_layers: [layer3]
page_type: concept
---

# Decomposition

`DecompositionComparator`は、複数目的を1つのスカラー値へ集約する処理を、`Decomposition`という差し替え可能なコンポーネントに委ねています。
MOEA/D{cite}`zhang2007moead`のようにサブ問題ごとの重みベクトルで探索を誘導する方式は、`Decomposition`と`DecompositionComparator`を組み合わせて実現します。

## Decompositionの役割

`Decomposition`が実装する必要があるメソッドは`aggregate(f, weights, ideal_point) -> np.ndarray`の1つだけです。`f`は形状`(N, n_obj)`の目的値行列、`weights`は形状`(n_obj,)`の非負の重みベクトル、`ideal_point`は形状`(n_obj,)`の理想点であり、形状`(N,)`のスカラー値を返します。スコアは低いほど良いという規約に統一されています。

`f`は、呼び出し側の`DecompositionComparator`が`direction`変換を適用済みの、最小化フレームである前提で渡されます。
独自の`Decomposition`を実装する際、`aggregate`の内部で改めて`direction`変換をする必要はありません。

## 組み込みDecomposition

いずれも{cite}`zhang2007moead`のMOEA/D論文がベースになっています。

| クラス | パラメータ | 特徴 |
|---|---|---|
| `WeightedSumDecomposition` | なし | `score = f @ weights`という線形重み付き和。最も単純だが非凸フロントの一部に到達できない |
| `TchebycheffDecomposition` | なし | `score = max_j(w_j * \|f_ij - z_j*\|)`というChebyshev距離。非凸フロントにも到達できる |
| `PBIDecomposition` | `theta=5.0` | `d1 + theta * d2`（重みベクトル方向への射影距離＋直交距離のペナルティ） |

`TchebycheffDecomposition`は、ゼロの重みをそのまま使うと退化してしまうため、内部で`1e-6`へ置換します（{cite}`zhang2007moead` Appendix Aの慣習）。

`PBIDecomposition`の`theta`は、収束（重みベクトル方向への射影距離`d1`）と多様性（直交距離`d2`へのペナルティ）のトレードオフを制御するパラメータです。
値が大きいほど、重みベクトルの方向から外れた解に強いペナルティがかかります。
`theta=5.0`は{cite}`zhang2007moead`の既定値です。

## DecompositionComparator

`DecompositionComparator(decomposition, weights, ideal_point=None, direction=None, *, eps_cv=0.0, eps_obj=1e-6)`は、[Comparator](comparators.md)のサブクラスとしてMOEA/D風の順位付けを実装します。

順序付けの規則は、実行可能性を優先（実行可能性優先{cite}`deb2000feasibility`）した上で、集約スコアの昇順で並べます。

`ideal_point`を省略（`None`）すると、`sort_population`は個体群の実行可能個体から理想点を動的に計算します。
一方、2点だけを比較する`compare`では、その2点の目的関数値の最小値をローカルな理想点近似として使います。
この2つの計算方法は完全には一致しない場合があります。

`weights`は非負の大きさのみを持ち、符号（最小化/最大化）は`direction`が表現するという役割分担になっています。
[Problem](problem.md)の`direction`/`weight`の役割分担と対応します。

## 独自Decompositionの実装方法

独自のスカラー化関数が必要な場合は、`Decomposition`を継承して`aggregate()`だけを実装すればよいです。
次の例は、重み付き積で集約する（全目的が正であることを前提とする）単純な実装です。

```python
import numpy as np
from saealib import Decomposition


class WeightedProductDecomposition(Decomposition):
    """Aggregation via weighted product (a simple example assuming all objectives are positive)."""

    def aggregate(self, f, weights, ideal_point):
        f = np.asarray(f, dtype=float)
        w = np.asarray(weights, dtype=float)
        return np.prod(np.abs(f - ideal_point + 1e-6) ** w, axis=1)
```

`f`は既に最小化フレームに変換済みの値として渡されるため、この実装内で`direction`を意識する必要はありません。

## 関連コンポーネント

- [Comparator](comparators.md)：`DecompositionComparator`が継承する基底クラス
- [Problem](problem.md)：`direction`/`weights`の役割分担

## 参照

- {py:class}`saealib.Decomposition`
- {py:class}`saealib.WeightedSumDecomposition`
- {py:class}`saealib.TchebycheffDecomposition`
- {py:class}`saealib.PBIDecomposition`
- {py:class}`saealib.DecompositionComparator`
