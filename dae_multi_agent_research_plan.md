# DAE を用いた Multi-Agent Advantage Estimation の研究方針

## 1. 研究の全体方針

本研究では，Direct Advantage Estimation（DAE）を協調型マルチエージェント強化学習（Cooperative MARL）へ拡張することを目的とする．

DAE は，従来の GAE のように value function から TD error を通じて advantage を推定するのではなく，advantage function を直接モデル化し，on-policy trajectory から学習する手法である．DAE の中心的な考え方は，advantage function が policy-centered な関数であり，その性質を利用することで return の分散を最小化しながら advantage を直接推定できる，という点にある．

一方，協調型 MARL では，報酬がチーム全体で共有されるため，単一の joint advantage をすべての agent にそのまま与えると credit assignment が曖昧になる．MAPPO では centralized critic と GAE により joint advantage を推定し，それを各 agent の policy update に共有して用いる．しかし，この設計では「どの agent のどの行動がチーム報酬にどれだけ貢献したか」を明示的には分解していない．

そこで本研究では，DAE を単に MAPPO に適用するだけでなく，multi-agent credit assignment を考慮した形へ段階的に拡張する．研究の流れは以下の 3 段階とする．

```text
DAE-MAPPO
  -> Factorized Multi-Agent DAE
      -> Counterfactual DAE / Ordered DAE
```

それぞれの目的は次の通りである．

1. **DAE-MAPPO**  
   MAPPO の GAE advantage を DAE advantage に置き換え，DAE が MARL の advantage estimator として機能するかを確認する．

2. **Factorized Multi-Agent DAE**  
   joint advantage を agent-wise advantage の和として表現し，各 agent に個別の advantage signal を与えることで credit assignment を改善する．

3. **Counterfactual DAE**  
   他 agent の action を条件にした counterfactual な agent-wise advantage を DAE で直接推定し，COMA 的な credit assignment を Q-function なしで実現する．順序付き分解に基づく変種（Ordered DAE，§4.8）も併せて検討する．

---

## 2. Phase 1: DAE-MAPPO

### 2.1 目的

最初の段階では，MAPPO の GAE を DAE に置き換える．

MAPPO では通常，centralized critic により value function を学習し，GAE によって joint advantage を推定する．

$$
\hat A_t^{\mathrm{GAE}}
\approx
A^\pi(s_t, \boldsymbol a_t)
$$

ここで，$s_t$ は global state または centralized observation，$\boldsymbol a_t = (a_t^1, \dots, a_t^N)$ は joint action である．

DAE-MAPPO では，この GAE advantage の代わりに，DAE により joint advantage を直接推定する．

$$
\hat A_\theta(s_t, \boldsymbol a_t)
\approx
A^\pi(s_t, \boldsymbol a_t)
$$

この段階の目的は，DAE が single-agent PPO だけでなく，MAPPO のような cooperative MARL setting でも advantage estimator として有効かを検証することである．

---

### 2.2 Joint DAE の設計

DAE-MAPPO では，joint action $\boldsymbol a_t$ を 1 つの大きな action とみなし，single-agent DAE を joint action space に適用する．

DAE loss は次のように定義する．

$$
\mathcal L_{\mathrm{J\text{-}DAE}} = \mathbb E \left[\left(\sum_{k=0}^{n-1}\gamma^k \left[r_{t+k} - \hat A_\theta(s_{t+k}, \boldsymbol a_{t+k})\right] + \gamma^n V_{\mathrm{target}}(s_{t+n}) - V_\phi(s_t)\right)^2 \right]
$$

ここで，

- $n$: backup horizon
- $V_{\mathrm{target}}$: target value network または old value function
- $V_\phi$: learned value function
- $\hat A_\theta$: DAE により直接学習される joint advantage

である．

DAE 論文と同様に，actor update では $\hat A_\theta$ に stop-gradient をかけて用いる．

---

### 2.3 Policy-centered constraint

single-agent DAE では，advantage function は policy-centered 条件を満たす．

$$
\sum_a \pi(a|s) A(s,a) = 0
$$

DAE-MAPPO の joint version では，この条件は joint policy に対して次のようになる．

$$
\sum_{\boldsymbol a} \pi(\boldsymbol a|s) \hat A_\theta(s,\boldsymbol a) = 0
$$

MAPPO の policy が factorized されている場合，

$$
\pi(\boldsymbol a|\boldsymbol o) = \prod_{i=1}^N \pi_i(a^i|o^i)
$$

であるため，理論上は joint action space 全体に対して expectation を取る必要がある．

ただし，joint action space は agent 数に対して指数的に大きくなるため，この centered constraint を厳密に計算するのは難しい．そのため，Phase 1 では以下のいずれかの実装を検討する．

#### 案 A: joint action が小さい環境で厳密に centering

小規模な matrix game や MPE の discrete action setting では，joint action space を列挙できる場合がある．その場合，

$$
\hat A_\theta(s,\boldsymbol a) = f_\theta(s,\boldsymbol a) - \sum_{\tilde{\boldsymbol a}} \pi_{\mathrm{old}}(\tilde{\boldsymbol a}|s)f_\theta(s,\tilde{\boldsymbol a})
$$

として，old policy に対して centered な advantage を構成する．

#### 案 B: sampled actions による Monte Carlo centering

joint action を複数サンプルし，

$$
\hat A_\theta(s,\boldsymbol a)
=
f_\theta(s,\boldsymbol a)
-
\frac{1}{K}
\sum_{j=1}^K
f_\theta(s,\tilde{\boldsymbol a}_j),
\quad
\tilde{\boldsymbol a}_j \sim \pi_{\mathrm{old}}(\cdot|s)
$$

と近似する．

ただし，variance と計算量が増える可能性がある．

#### 案 C: Phase 1 では centering を弱く扱う

Phase 1 はあくまで DAE の MARL 適用可能性の確認と位置づけ，厳密な joint centering は行わず，後続の factorized DAE で agent-wise centering を本格的に導入する．

---

### 2.4 Actor update

MAPPO の actor update では，各 agent $i$ の PPO ratio を

$$
r_t^i(\theta_i)
=
\frac{
\pi_{\theta_i}^i(a_t^i|o_t^i)
}{
\pi_{\mathrm{old}}^i(a_t^i|o_t^i)
}
$$

とする．

DAE-MAPPO では，各 agent が同じ joint DAE advantage を用いる．

$$
L_i^{\mathrm{J\text{-}DAE}}
=
\mathbb E
\left[
\min
\left(
 r_t^i \operatorname{sg}(\hat A_\theta(s_t,\boldsymbol a_t)),
 \operatorname{clip}(r_t^i,1-\epsilon,1+\epsilon)
 \operatorname{sg}(\hat A_\theta(s_t,\boldsymbol a_t))
\right)
\right]
$$

ここで $\operatorname{sg}(\cdot)$ は stop-gradient を表す．

---

### 2.5 この段階で確認すること

DAE-MAPPO は新規性というよりも，後続の factorized DAE のための重要な baseline である．この段階では以下を確認する．

- GAE を DAE に置き換えても安定して学習できるか
- DAE loss が MARL の shared reward setting で崩壊しないか
- DAE advantage の scale が PPO update に適しているか
- GAE と比較して sample efficiency が改善するか
- DAE に必要な batch size や network capacity はどの程度か

---

## 3. Phase 2: Factorized Multi-Agent DAE

### 3.1 目的

Phase 2 では，joint advantage を agent-wise advantage に分解する．

MAPPO の問題点は，すべての agent が同じ scalar advantage を使って policy update されることである．

$$
\hat A_t^{\mathrm{joint}}
$$

この場合，チームとしての結果が良かったか悪かったかは分かるが，各 agent の行動がどの程度貢献したかは分からない．

そこで，Factorized Multi-Agent DAE では，joint advantage を agent-wise advantage の和として表現する．

$$
\hat A_{\mathrm{tot}}(s_t,\boldsymbol a_t)
=
\sum_{i=1}^{N}
\hat A_i(s_t,o_t^i,a_t^i)
$$

または centralized input を強く使う場合は，

$$
\hat A_{\mathrm{tot}}(s_t,\boldsymbol a_t)
=
\sum_{i=1}^{N}
\hat A_i(s_t,\boldsymbol o_t,a_t^i)
$$

とする．

これにより，DAE の direct advantage estimation と multi-agent credit assignment を統合する．

---

### 3.2 Factorized DAE loss

Factorized DAE では，DAE loss の中で reward residual を agent-wise advantage の和により説明する．

$$
\mathcal L_{\mathrm{F\text{-}DAE}}
=
\mathbb E
\left[
\left(
\sum_{k=0}^{n-1}
\gamma^k
\left[
 r_{t+k}
 -
 \sum_{i=1}^{N}
 \hat A_i(s_{t+k},o_{t+k}^i,a_{t+k}^i)
\right]
+
\gamma^n V_{\mathrm{target}}(s_{t+n})
-
V_\phi(s_t)
\right)^2
\right]
$$

この loss は，trajectory 上の return を，value function と agent-wise advantage の時系列和で説明するように学習する．

DAE の視点では，これは「各時刻の reward から agent-wise advantage contribution を引いた transformed return の分散を最小化する」ことに対応する．

---

### 3.3 Agent-wise centered constraint

Factorized DAE の重要な設計は，各 agent advantage をその agent の policy に対して centered にすることである．

各 agent $i$ について，

$$
\sum_{a^i}
\pi_i(a^i|o^i)
\hat A_i(s,o^i,a^i)
=0
$$

を課す．

実装では raw advantage head $f_i$ を用意し，old policy $\mu_i$ に対して centering する．

$$
\hat A_i(s,o^i,a^i)
=
f_i(s,o^i,a^i)
-
\sum_{\tilde a^i}
\mu_i(\tilde a^i|o^i)
 f_i(s,o^i,\tilde a^i)
$$

ここで $\mu_i$ は rollout を生成した old policy である．

PPO update 中に policy $\pi_i$ は変化するため，centering は current policy ではなく old policy $\mu_i$ に対して行う方が安定である．これは DAE 論文の PPO 統合において，sampling policy $\mu$ を固定して advantage を centered にする設計と対応する．

---

### 3.4 Actor update: agent-specific advantage

Factorized DAE では，各 agent の actor update に，その agent 専用の advantage を使う．

$$
L_i^{\mathrm{F\text{-}DAE}}
=
\mathbb E
\left[
\min
\left(
 r_t^i \operatorname{sg}(\hat A_i(s_t,o_t^i,a_t^i)),
 \operatorname{clip}(r_t^i,1-\epsilon,1+\epsilon)
 \operatorname{sg}(\hat A_i(s_t,o_t^i,a_t^i))
\right)
\right]
$$

これにより，agent $i$ の policy は，joint advantage ではなく，agent $i$ に割り当てられた advantage signal に基づいて更新される．

この設計の狙いは，MAPPO のように全 agent が同じ signal で更新される状況を避け，より明示的な credit assignment を導入することである．

---

### 3.5 分解の一意性と収束先の特徴付け

shared reward のみから agent-wise advantage を学習する場合，一見すると分解は非一意に思える．しかし，方策が factorized（agent 間で独立）であるとき，**加法的な centered クラスに限れば分解は一意であり，さらに収束先を明示的に特徴付けられる**．以下の 3 つの命題は証明スケッチに基づく予想であり，論文化に向けて形式証明を行う（仮定：discrete action，factorized policy，centering は rollout policy $\mu$ に対して，population limit，到達可能な state-action 対上）．

#### 命題 A（分解の一意性）

各 $g_i(s,a^i)$ が $\mu_i$-centered で，到達可能な組に対して $\sum_i g_i = 0$ ならば，$g_i \equiv 0$ である．

証明スケッチ：$a^i$ を固定し他 agent の action について $\mu^{-i}$ で期待値を取ると，方策の独立性と centering により $j \neq i$ の項がすべて消え，$g_i(s,a^i) = 0$ を得る．

したがって F-DAE の解空間に null 方向は存在せず，**identifiability のための正則化は本質的には不要**である．

#### 命題 B（収束先 = marginal advantage）

F-DAE loss の population minimizer は

$$
\hat A_i^*(s,o^i,a^i)
=
\mathbb E_{\boldsymbol a^{-i}\sim\mu^{-i}}
\left[
A^\mu(s,\boldsymbol a)
\,\middle|\,
a^i
\right]
$$

すなわち **marginal advantage**（COMA counterfactual advantage の $\boldsymbol a^{-i}$ に関する期待値）である．

証明スケッチ：centered 関数は過去の履歴に条件付けても平均 0 であるため，DAE loss を展開したときの時刻間交差項が消え，最小化は訪問分布で重み付けた $L^2$ 射影に帰着する（DAE 論文 Theorem 1 の議論の部分クラス版）．積測度の下での加法的クラスへの $L^2$ 射影は functional ANOVA により $\sum_i \mathbb E[A \mid a^i]$ で与えられる．

#### 命題 C（agent 別勾配の不偏性）

$\pi = \mu$ において，marginal advantage を用いた agent 別 policy gradient は，joint advantage を用いた真の gradient と一致する．

$$
\mathbb E_{\boldsymbol a \sim \mu}
\left[
\nabla \log \pi_i(a^i|o^i)\, A(s,\boldsymbol a)
\right]
=
\mathbb E_{a^i \sim \mu_i}
\left[
\nabla \log \pi_i(a^i|o^i)\, \hat A_i^*(s,o^i,a^i)
\right]
$$

証明スケッチ：方策の独立性より $\boldsymbol a^{-i}$ を先に周辺化すればよい．

命題 A–C により，F-DAE-agent は「一意な固定点を持ち，固定点における agent 別勾配が不偏」という理論的裏付けを得る．これが Hypothesis 2 の根拠となる．

#### 正則化の位置付け

上記より，L2 正則化

$$
\mathcal L_{\mathrm{reg}}
=
\alpha
\sum_{i=1}^{N}
\mathbb E
\left[
\hat A_i(s,o^i,a^i)^2
\right]
$$

は identifiability の装置ではなく，有限サンプル・学習途中における安定化として位置付ける（有無を ablation で比較する）．なお，CF-DAE では事情が逆転し，正則化が本質的に必要になる（§4.5）．

#### 表現力の限界（Phase 3 への動機）

一方で，加法的クラスは agent 間の相互作用項を表現できない．例えば 2 agent の一致（XOR 型）ゲームでは，対称方策点において marginal advantage が恒等的に 0 となり，F-DAE の学習信号は「分散ゼロのゼロ勾配」に退化して対称点から抜け出せない．この表現力の限界が，Phase 3（counterfactual / ordered 分解）の直接の動機である．

---

### 3.6 入力設計

Factorized DAE の agent advantage head には複数の入力設計が考えられる．

#### Local-input version

$$
\hat A_i(o^i,a^i)
$$

各 agent の local observation と action のみを使う．decentralized execution と整合的だが，credit assignment の情報が不足する可能性がある．

#### Centralized-state version

$$
\hat A_i(s,o^i,a^i)
$$

global state と local observation/action を使う．centralized training を活用できるため，最初はこちらが有力である．

#### Joint-observation version

$$
\hat A_i(\boldsymbol o,a^i)
$$

global state がない環境では，joint observation を centralized input として使う．

推奨する初期設計は，

$$
\hat A_i(s,o^i,a^i)
$$

である．

---

### 3.7 比較実験

Factorized DAE の有効性を見るため，以下を比較する．

| Method | Advantage estimator | Actor update |
|---|---|---|
| MAPPO | GAE joint advantage | same advantage for all agents |
| J-DAE-MAPPO | DAE joint advantage | same advantage for all agents |
| F-DAE-global | factorized DAE | summed advantage for all agents |
| F-DAE-agent | factorized DAE | agent-specific advantage |
| F-DAE no-centering | factorized DAE without centering | agent-specific advantage |
| F-DAE no-reg | factorized DAE without regularization | agent-specific advantage |

特に重要なのは，以下の比較である．

1. **MAPPO vs J-DAE-MAPPO**  
   DAE が GAE の代替として機能するか．

2. **J-DAE-MAPPO vs F-DAE-agent**  
   factorization による credit assignment が有効か．

3. **F-DAE-agent vs F-DAE no-centering**  
   DAE の centered constraint が性能に寄与しているか．

---

## 4. Phase 3: Counterfactual DAE

### 4.1 目的

Factorized DAE では，agent $i$ の advantage を主に

$$
\hat A_i(s,o^i,a^i)
$$

として表現する．

しかし，協調型 MARL では，agent の貢献は他 agent の action に強く依存する．例えば，ある agent の action が有効かどうかは，他 agent がどの行動を選んだかによって変わる．

そこで，Counterfactual DAE では，agent $i$ の advantage を他 agent の action を条件にして表現する．

$$
\hat A_i(s,\boldsymbol a^{-i},a^i)
$$

ここで，$\boldsymbol a^{-i}$ は agent $i$ 以外の action を表す．

この設計により，COMA の counterfactual advantage に近い credit assignment を，Q-function を経由せずに DAE で直接学習することを目指す．

---

### 4.2 COMA との関係と推定対象の再定式化

COMA では，agent $i$ の counterfactual advantage は次のように定義される．

$$
A_i^{\mathrm{COMA}}(s,\boldsymbol a)
=
Q(s,\boldsymbol a)
-
\sum_{\tilde a^i}
\pi_i(\tilde a^i|o^i)
Q(s,(\boldsymbol a^{-i},\tilde a^i))
$$

これは，他 agent の action $\boldsymbol a^{-i}$ を固定した上で，agent $i$ の action が baseline と比べてどれだけ良かったかを測る．

一方，Counterfactual DAE では Q-function を学習せず，$\hat A_i(s,\boldsymbol a^{-i},a^i)$ を直接モデル化する．

ただし，ここで重要な注意点がある．**COMA advantage の和は joint advantage に一致しない**．

$$
\sum_{i=1}^N A_i^{\mathrm{COMA}}(s,\boldsymbol a)
=
N \cdot A(s,\boldsymbol a)
-
\sum_{i=1}^N
\mathbb E_{\tilde a^i \sim \pi_i}
\left[
A(s,(\boldsymbol a^{-i},\tilde a^i))
\right]
\neq
A(s,\boldsymbol a)
$$

一方，CF-DAE loss（§4.4）は $\sum_i \hat A_i$ が joint advantage（reward residual）を説明するように働くため，収束先の候補集合に COMA advantage の組は含まれない．したがって「COMA の counterfactual advantage そのものを Q なしで直接推定する」という主張は成立せず，本手法の推定対象は次のように再定義する．

```text
CF-DAE の推定対象:
joint advantage の counterfactual 条件付き centered 分解

A(s, a) = Σ_i Â_i(s, a^{-i}, a^i)
s.t.  Σ_{a^i} π_i(a^i|o^i) Â_i(s, a^{-i}, a^i) = 0
```

この再定式化はむしろ望ましい性質を持つ．分解の和が joint advantage に厳密に一致することは，Kakade–Langford 型の policy improvement 論法と直結し，「agent 別更新の総和が joint な改善に対応する」という解釈を与えるからである．COMA との対応は期待値レベルで残る：F-DAE の固定点（marginal advantage，§3.5 命題 B）は COMA advantage の $\boldsymbol a^{-i}$ に関する期待値であり，CF-DAE はこれを $\boldsymbol a^{-i}$ 条件付きに精細化したものと位置付けられる．

```text
COMA:   Q-function から counterfactual advantage を構成する
F-DAE:  counterfactual advantage の a^{-i} 期待値（marginal advantage）を直接推定する
CF-DAE: joint advantage の counterfactual 条件付き centered 分解を直接推定する
```

---

### 4.3 Counterfactual centered constraint

Counterfactual DAE では，各 agent advantage を，他 agent の action を固定した条件下で centered にする．

$$
\sum_{a^i}
\pi_i(a^i|o^i)
\hat A_i(s,\boldsymbol a^{-i},a^i)
=0
$$

実装では raw head $f_i$ に対して，

$$
\hat A_i(s,\boldsymbol a^{-i},a^i)
=
f_i(s,\boldsymbol a^{-i},a^i)
-
\sum_{\tilde a^i}
\mu_i(\tilde a^i|o^i)
 f_i(s,\boldsymbol a^{-i},\tilde a^i)
$$

とする．

ここでも $\mu_i$ は rollout を生成した old policy である．

---

### 4.4 Counterfactual DAE loss

Counterfactual DAE では，reward residual を counterfactual agent-wise advantage の和で説明する．

$$
\mathcal L_{\mathrm{CF\text{-}DAE}}
=
\mathbb E
\left[
\left(
\sum_{k=0}^{n-1}
\gamma^k
\left[
 r_{t+k}
 -
 \sum_{i=1}^{N}
 \hat A_i(s_{t+k},\boldsymbol a_{t+k}^{-i},a_{t+k}^i)
\right]
+
\gamma^n V_{\mathrm{target}}(s_{t+n})
-
V_\phi(s_t)
\right)^2
\right]
$$

actor update では，agent $i$ に対応する counterfactual advantage を用いる．

$$
L_i^{\mathrm{CF\text{-}DAE}}
=
\mathbb E
\left[
\min
\left(
 r_t^i \operatorname{sg}(\hat A_i(s_t,\boldsymbol a_t^{-i},a_t^i)),
 \operatorname{clip}(r_t^i,1-\epsilon,1+\epsilon)
 \operatorname{sg}(\hat A_i(s_t,\boldsymbol a_t^{-i},a_t^i))
\right)
\right]
$$

---

### 4.5 表現力と分解の非一意性，最小ノルム正則化

CF-DAE の関数クラスの性質は，F-DAE（§3.5）と対照的である．

#### 表現力は完全

クラス $\mathcal F_{\mathrm{cf}} = \left\{\sum_i f_i(s,\boldsymbol a^{-i},a^i) : f_i \text{ は centered}\right\}$ は joint advantage を厳密に表現できる．構成例として，telescoping 分解

$$
f_i
=
\mathbb E\left[A \mid \boldsymbol a^{\le i}\right]
-
\mathbb E\left[A \mid \boldsymbol a^{< i}\right]
$$

（残りの agent の action は $\mu$ で周辺化する）は各項が centering 条件を満たし，和が $A$ に厳密に一致する（§4.8 の順序付き分解）．

#### しかし分解は一意でない

$\mathcal F_{\mathrm{cf}}$ の null space は非自明である．反例：2 agent，binary action，一様方策のとき

$$
\delta(a^1,a^2) = (-1)^{a^1+a^2}
$$

は両 agent の centering を満たし，$(\hat A_1 + \delta,\ \hat A_2 - \delta)$ も同じ和を与える．null 成分 $\delta$ は $a^i$ に依存するため，**DAE loss（和）には影響しないが，agent 別の policy gradient を実際に変える**．つまり CF-DAE では，actor に渡る credit 信号が loss だけからは決まらない．これが Phase 3 固有の本質的課題であり，F-DAE では起きない（§3.5 命題 A）．

#### 最小ノルム分解としての正則化

そこで L2 正則化を identifiability の装置として再解釈する．$A$ の centered 分解全体はアフィン空間であり，その上で $\sum_i \mathbb E_\mu[f_i^2]$ は強凸であるため，**最小ノルム centered 分解は一意に定まる**．

$$
(\hat A_1^*, \dots, \hat A_N^*)
=
\mathop{\arg\min}_{\substack{\sum_i f_i = A \\ f_i:\ \text{centered}}}
\sum_{i=1}^N
\mathbb E_\mu\left[f_i(s,\boldsymbol a^{-i},a^i)^2\right]
$$

「CF-DAE + L2 正則化（$\alpha \to 0$ の極限）」の推定対象をこの最小ノルム分解として定義することで，CF-DAE の推定対象が well-defined になる．最小ノルム分解が具体的にどのような credit を与えるか（例：対称な agent への均等配分性）の解析は理論課題として残す．一意性を構造的に保証する代替案として，順序付き分解（§4.8）がある．

---

### 4.6 Counterfactual DAE の利点

Counterfactual DAE には以下の利点がある．

#### 1. 他 agent の action を考慮した credit assignment

$$
\hat A_i(s,\boldsymbol a^{-i},a^i)
$$

により，agent $i$ の action の価値を，他 agent の actual action を条件に評価できる．

#### 2. Q-function を必要としない

COMA のように centralized Q-function を学習する必要がない．DAE の枠組みにより，advantage を直接推定する．

#### 3. DAE の centered constraint と自然に接続する

他 agent の action を固定した条件下で，agent $i$ の action に関して policy-centered にすることで，counterfactual baseline を自然に構成できる．

#### 4. 新規性が明確

「counterfactual advantage を Q から構成する」のではなく，「joint advantage の counterfactual 条件付き centered 分解を直接推定する」点が新規性になる（§4.2 の再定式化）．

---

### 4.7 課題

Counterfactual DAE には以下の課題がある．

#### 入力次元の増加

$\boldsymbol a^{-i}$ を入力するため，agent 数が増えると入力が大きくなる．

#### action encoding の設計

discrete action なら one-hot encoding が使えるが，連続 action では扱いが難しい．

#### 分解の非一意性

CF クラスの null space は非自明であり（§4.5），agent 別の credit 信号は DAE loss だけからは定まらない．最小ノルム正則化（§4.5）か，順序付き分解による構造的な一意化（§4.8）が必要になる．なお，F-DAE ではこの問題は起きない（§3.5 命題 A）．

#### scalability

全 agent について $\hat A_i(s,\boldsymbol a^{-i},a^i)$ を計算するため，agent 数が大きい環境では計算量が増える．

---

### 4.8 変種：順序付き分解による Ordered DAE（O-DAE）

CF-DAE の非一意性（§4.5）を，正則化ではなく関数クラスの構造によって解決する変種として，順序付き分解に基づく Ordered DAE を併置する．以下の性質は証明スケッチに基づく予想であり，形式証明は今後の課題である．

#### 定義

agent の順序（permutation）$\sigma = (1, \dots, N)$ を固定し，agent $i$ の advantage head を**先行 agent の action のみ**に条件付ける．

$$
\hat A_i(s, \boldsymbol a^{<i}, a^i)
=
f_i(s, \boldsymbol a^{<i}, a^i)
-
\sum_{\tilde a^i}
\mu_i(\tilde a^i|o^i)
f_i(s, \boldsymbol a^{<i}, \tilde a^i)
$$

centering は自 agent の action 空間の和のみで厳密に計算できる（CF-DAE と同じく tractable）．DAE loss は CF-DAE（§4.4）と同形で，residual に $\sum_i \hat A_i(s,\boldsymbol a_t^{<i},a_t^i)$ を用いる．actor update は agent $i$ に $\operatorname{sg}(\hat A_i(s,\boldsymbol a_t^{<i},a_t^i))$ を渡す．

#### 理論的性質：完全かつ一意

multi-agent advantage decomposition lemma（Kuba et al., HATRPO/HAPPO；MAT でも利用）より，joint advantage は順序付きの和に厳密に分解できる．

$$
A(s,\boldsymbol a)
=
\sum_{i=1}^N
A^{i}(s, \boldsymbol a^{<i}, a^i),
\qquad
A^{i}
:=
\mathbb E\left[A \mid \boldsymbol a^{\le i}\right]
-
\mathbb E\left[A \mid \boldsymbol a^{< i}\right]
$$

したがって順序付きクラス $\mathcal F_{\mathrm{ord}}$ は表現力が完全である．さらに分解は一意である：$\sum_i f_i = 0$ のとき，$\boldsymbol a^{\le k}$ で条件付き期待値を取ると $i > k$ の項が centering により消えるため，$\sum_{i \le k} f_i = 0$ が全ての $k$ で成り立ち，帰納的に $f_k \equiv 0$ を得る．DAE loss は和のみに依存するため，population minimizer において各 head は decomposition lemma の項 $A^i$ に一致する．**推定対象が正則化なしで一意に定まる**．また，agent 別勾配の不偏性（§3.5 命題 C）は O-DAE の信号 $A^i$ でも同じ論法で成り立つ．

3 つの関数クラスは入れ子関係にある．

$$
\mathcal F_{\mathrm{add}} \subset \mathcal F_{\mathrm{ord}} \subset \mathcal F_{\mathrm{cf}}
$$

| クラス | 表現力 | 分解の一意性 | 推定対象 |
|---|---|---|---|
| F-DAE（additive） | 加法近似のみ | 一意 | marginal advantage（§3.5 命題 B） |
| O-DAE（ordered） | 完全 | 一意 | decomposition lemma の各項 $A^i$ |
| CF-DAE（counterfactual） | 完全 | 非一意 | 最小ノルム centered 分解（§4.5） |

O-DAE は「完全な表現力」と「分解の一意性」を両立する中間クラスであり，理論面では最も筋の良い設計である．

#### 順序依存性とランダム順列，Shapley 分解

固定順序では，agent 1 は常に marginal な（弱い）信号を，後段の agent ほど条件付きの（強い）信号を受ける非対称性がある．緩和策として，rollout（または minibatch）ごとにランダムな順列 $\sigma$ を用いる（MAT と同様）．このとき head の入力に「どの agent が先行するか」を表す mask を含める．

ランダム順列で期待値を取った agent $i$ の credit は，coalition game

$$
v_s(C; \boldsymbol a_C) = \mathbb E_{\boldsymbol a_{\bar C} \sim \mu}\left[A(s,\boldsymbol a) \mid \boldsymbol a_C\right]
$$

の **Shapley value** に一致し，efficiency（$\sum_i \phi_i = A$）を満たす．つまり permutation-averaged O-DAE は「joint advantage の Shapley 分解の直接推定」と解釈できる．Shapley Q-value 系の既存研究（SQDDPG，Shapley counterfactual credit など）が Q-function を経由するのに対し，本手法は advantage を直接推定する点で差別化される．

#### HAPPO との関係

HAPPO は同じ分解を importance ratio の積 $M^{1:i}$ によって Monte Carlo 的に構成する（agent の順番が進むほど分散が増大する）．O-DAE は各項 $A^i$ を関数近似で直接推定する低分散の代替と位置付けられる．O-DAE head を HAPPO の逐次更新スキームに組み込み，monotonic improvement 保証と接続する拡張は将来課題とする．

#### 実装方針

- 現行の advantage table 実装（centralized features から $N \times |\mathcal A|$ テーブル）を拡張し，先行 agent の action の one-hot（後続 agent 分は zero）と先行 mask を条件付け入力に追加する
- agent 方向に causal mask をかけた attention（MAT decoder 型）により，全 agent 分の head を 1 パスで計算する設計が自然である
- 比較実験には F-DAE / CF-DAE / O-DAE（固定順・ランダム順列）を含め，一致（XOR 型）ゲームで「F-DAE は対称点で信号消失，O-DAE / CF-DAE は条件付き信号により対称性を破れる」ことを確認する

---

## 5. 実装上の共通注意点

### 5.1 stop-gradient

actor update に用いる advantage estimate は stop-gradient する．

$$
\operatorname{sg}(\hat A)
$$

DAE head は DAE loss によって更新し，actor loss から advantage head へ勾配を流さない．

---

### 5.2 old policy による centering

PPO update 中に policy は変化するため，centering は current policy ではなく rollout policy，すなわち old policy $\mu$ に対して行う．

$$
\hat A_i
=
f_i
-
\mathbb E_{a^i\sim \mu_i}
[f_i]
$$

これにより，DAE loss と PPO objective の on-policy 性が保たれやすくなる．

---

### 5.3 advantage normalization

DAE advantage は GAE と scale が異なる可能性がある．PPO update の安定化のため，以下を比較する．

- batch 全体で advantage normalization
- agent ごとに advantage normalization
- normalization なし

Factorized DAE では，agent ごとの scale が異なる可能性があるため，agent-wise normalization も有力である．

---

### 5.4 batch size

DAE は advantage function を network で直接近似するため，GAE よりも batch size や network capacity の影響を受けやすい可能性がある．

そのため，MARL でも十分に大きな rollout batch を使うことが重要である．

---

### 5.5 value loss の扱い

DAE loss には value function $V_\phi$ が含まれるが，初期実装では MAPPO の value loss を補助的に残すことも検討する．

候補は以下である．

#### DAE loss のみ

$$
\mathcal L
=
\mathcal L_{\pi}
+
\beta_{\mathrm{DAE}}\mathcal L_{\mathrm{DAE}}
+
\beta_{\mathrm{ent}}\mathcal L_{\mathrm{ent}}
$$

#### DAE loss + value loss

$$
\mathcal L
=
\mathcal L_{\pi}
+
\beta_{\mathrm{DAE}}\mathcal L_{\mathrm{DAE}}
+
\beta_V\mathcal L_V
+
\beta_{\mathrm{ent}}\mathcal L_{\mathrm{ent}}
$$

最初は安定性を重視し，DAE loss + value loss を使うのが安全である．

---

<!-- ## 6. 実験計画

### 6.1 推奨環境

最初は軽量な discrete action cooperative MARL 環境を用いる．

候補：

- Matrix coordination game
- MPE simple_spread
- MPE simple_reference
- Level-Based Foraging
- SMAC small maps
- Hanabi small setting

DAE は centered constraint の計算が必要なため，まずは discrete action 環境が望ましい．

---

### 6.2 比較手法

最低限，比較すべき手法は以下である．

1. MAPPO + GAE
2. MAPPO + Joint DAE
3. MAPPO + Factorized DAE
4. MAPPO + Factorized DAE without centering
5. MAPPO + Factorized DAE without regularization
6. MAPPO + Counterfactual DAE

---

### 6.3 評価指標

評価指標は以下を用いる．

- episodic return
- sample efficiency
- final performance
- win rate，SMAC などの場合
- advantage estimate の variance
- agent-wise advantage の分散・scale
- credit assignment の interpretability

可能であれば，controlled environment で真の credit が分かるタスクを作り，agent-wise advantage が妥当な分解になっているかを確認する．

--- -->

## 7. 研究仮説

本研究の主要仮説は以下である．

### Hypothesis 1: DAE is a viable alternative to GAE in MAPPO

DAE により joint advantage を直接推定することで，GAE よりも低分散かつ安定した policy update が可能になる．

### Hypothesis 2: Factorized DAE improves credit assignment

joint advantage を agent-wise advantage に分解し，各 agent の policy update に agent-specific advantage を用いることで，MAPPO の credit assignment を改善できる．

### Hypothesis 3: Policy-centered constraints are important in multi-agent DAE

agent-wise centered constraint を課すことで，DAE の理論的性質を保ち，学習の安定性と性能が改善する．

### Hypothesis 4: Counterfactual DAE improves coordination-sensitive credit assignment

他 agent の action を条件にした counterfactual advantage を直接推定することで，協調依存の強い環境においてより適切な credit assignment が可能になる．

### Hypothesis 5: Ordered decomposition resolves the identifiability–expressiveness trade-off

順序付き分解（O-DAE，§4.8）は表現力の完全性と分解の一意性を両立し，正則化に依存せずに CF-DAE と同等以上の credit assignment を実現できる．

---

## 8. まとめ

本研究では，DAE を cooperative MARL に拡張するため，以下の段階的な設計を採用する．

```text
1. DAE-MAPPO
   - GAE を DAE に置き換える
   - joint advantage を直接推定する
   - MARL setting で DAE が機能するか確認する

2. Factorized Multi-Agent DAE
   - joint advantage を agent-wise advantage の和に分解する
   - 各 agent advantage に policy-centered constraint を課す
   - agent-specific advantage により policy update を行う
   - MAPPO の credit assignment 問題を改善する

3. Counterfactual DAE / Ordered DAE
   - 他 agent の action を条件にした advantage を直接推定する
   - CF-DAE: joint advantage の counterfactual 条件付き centered 分解を推定し，
     最小ノルム正則化により一意化する
   - O-DAE: 順序付き分解により一意かつ完全な分解を直接推定する
     （ランダム順列により Shapley 分解へ拡張）
   - COMA 的な counterfactual credit assignment を Q-function なしで実現する
```

最初のゴールは，MAPPO ベースで Factorized Multi-Agent DAE を安定に実装し，GAE および Joint DAE に対する有効性を示すことである．その後，Counterfactual DAE / Ordered DAE へ拡張し，DAE の direct advantage estimation と counterfactual credit assignment を統合する．CF-DAE と O-DAE は表現力と一意性の観点で相補的であり，両者を比較することで agent-wise 分解の設計原理自体を検証する．
