論文・計画・現在の実装（onpolicy 側の diff）を一通り読みました。結論から言うと、段階的な設計方針と CF-DAE の方向性には賛成で、新規性も本物だと思います。ただし、理論面で計画の主張を左右する点が 3 つあり、特に「CF-DAE = COMA を Q なしで実現」という中心的主張はこのままでは成立しません。一方で、計画が心配している問題（Phase 2 の非一意性）は実は起きない、という良い方向の発見もあります。順に説明します。

1. Phase 2 (F-DAE) は計画が思っているより性質が良い
§3.5 で「分解の非一意性」を懸念していますが、加法的クラスに限れば分解は一意です。各 $g_i(s, a^i)$ が $\pi_i$-centered で $\sum_i g_i = 0$ なら、$a^i$ で条件付き期待値を取ると（方策が独立なので）$g_i(s,a^i) = 0$ が直ちに出ます。つまり null space は自明で、正則化は identifiability のためには本質的に不要です（有限サンプルでの安定化としては依然有用）。

さらに重要なのは収束先が特徴付けられることです。DAE の損失は centered 関数に対して時刻間の交差項が消えるので（$\mathbb{E}[\hat A_{t'} \mid \text{過去}] = 0$）、部分クラスでの最小化は訪問分布重み付き $L^2$ での射影になります。加法的クラスへの射影は functional ANOVA そのもので、解は

$$
\hat A_i^*(s, a^i) = \mathbb{E}_{\boldsymbol a^{-i} \sim \mu^{-i}}\left[ A^\pi(s, \boldsymbol a) ,\middle|, a^i \right]
$$

すなわち marginal advantage（COMA advantage の $\boldsymbol a^{-i}$ に関する期待値） です。そしてこの信号による agent 別 policy gradient は、joint advantage を使った真の勾配と（$\pi = \mu$ において）厳密に一致します：$\mathbb{E}{\boldsymbol a}[\nabla \log \pi_i(a^i) , A(s,\boldsymbol a)] = \mathbb{E}{a^i}[\nabla \log \pi_i(a^i) , \hat A_i^*(s,a^i)]$。

つまり「F-DAE-agent は一意な固定点を持ち、その固定点での agent 別勾配は不偏」という命題が立てられます。これは Hypothesis 2 の理論的裏付けとしてそのまま論文に書けるレベルの結果なので、形式化する価値が高いです。

2. CF-DAE の中心的主張は再定式化が必要
§4.2 の「COMA の counterfactual advantage を直接推定する」は成り立ちません。理由は単純で、COMA advantage の和は joint advantage に一致しないからです：

$$
\sum_i A_i^{\mathrm{COMA}} = N \cdot A - \sum_i \mathbb{E}_{\tilde a^i}[A \mid \boldsymbol a^{-i}] \neq A
$$

一方 CF-DAE の損失は $\sum_i \hat A_i \to A$ となるよう働くので、収束先の候補集合に COMA advantage の組は含まれません。学習されるのは「joint advantage の counterfactual 条件付き centered 分解のどれか」です。主張は「COMA を Q なしで再現」ではなく、**「joint advantage を、他 agent の実際の行動に条件付いた agent 別 centered 成分へ直接分解して学習する」**と再定義すべきです。個人的にはこちらの方がむしろ筋が良いと思います — 分解の和が joint advantage に一致することは、joint policy improvement の保証（Kakade–Langford 型）と直結するからです。

そして F-DAE と対照的に、CF-DAE クラスでは表現力と一意性が逆転します：

表現力は十分：telescoping 分解 $f_i = \mathbb{E}[A \mid a^{\le i}] - \mathbb{E}[A \mid a^{<i}]$ は各項が CF-DAE の centering 条件を満たし、和は厳密に $A$ になる（構成的証明）。
しかし null space が非自明：例えば 2 agent・binary action・一様方策で $\delta(a^1,a^2) = (-1)^{a^1+a^2}$ は両方の centering を満たしつつ和に寄与しない。null 成分は $a^i$ に依存するため agent 別勾配を実際に変えます。つまり CF-DAE でこそ正則化が本質的で、§3.5 の懸念は Phase 2 ではなく Phase 3 の問題です。L2 正則化を入れるなら「最小ノルム centered 分解」として解を特徴付けられる可能性があり、これも理論ネタになります。
「F-DAE：一意だが加法的で表現力不足 ↔ CF-DAE：表現力完全だが分解が不定」という対比は、論文の綺麗な軸になります。

3. 第三の選択肢：順序付き分解（強く推奨）
HATRPO/HAPPO の multi-agent advantage decomposition lemma（Kuba et al. 2022、MAT も利用）に対応する ordered CF-DAE を検討してください：$f_i(s, \boldsymbol a^{<i}, a^i)$、centering は $\pi_i$ に対して。このクラスは

表現力完全（telescoping 分解がまさにこの形）かつ 分解が一意（$a^{\le k}$ で条件付けする帰納法で null space が自明と示せる）
推定対象が $A^i(s,\boldsymbol a^{<i},a^i) = \mathbb{E}[A|a^{\le i}] - \mathbb{E}[A|a^{<i}]$ と明確
という、F-DAE と CF-DAE の良いとこ取りになっています。順序依存性は MAT 同様ランダム順列で緩和できます。CF-DAE 本体と並ぶ variant として持っておくと、理論の穴（非一意性）を突かれたときの受け皿になります。

4. Hypothesis 4 の killer experiment：一致（XOR）ゲーム
2 agent が同じ行動を選ぶと報酬 1、違うと 0 のゲームでは、対称方策点で marginal advantage が恒等的にゼロになります。つまり F-DAE は「分散ゼロのゼロ勾配」を出して saddle から抜けられないのに対し、CF-DAE は実際の $a^{-i}$ に条件付くので O(1) の信号を出せます。これは F-DAE と CF-DAE を最小構成で分離する診断環境です。現在 §6（実験計画）がコメントアウトされていますが、matrix game のパートは復活させて、この環境を Phase 3 の動機付けの中心に据えることを勧めます。SMAC だけだと credit assignment の改善が本当に効いたのか判別しにくいです。

5. Narrative の提案：centering の tractability
計画では centering を Phase 1 の障害として扱っていますが（案A/B/C）、視点を変えると：exact centering が指数的に高くつくのは joint パラメータ化だけで、F-DAE / CF-DAE / ordered はどれも自分の行動空間 $|\mathcal A_i|$ の和だけで厳密に centering できる。つまり「MARL の構造こそが DAE のボトルネックを解消する」という筋で、factorization が credit assignment のためだけの装置ではなくなります。Phase 1 の joint centering に深入りする価値は低く、案 C（小規模環境で最小限）で十分だと思います。

実装について（diff を見た範囲）
現在の実装は実は J-DAE ではなく F-DAE-global です。r_actor_critic.py の adv_out は $(N \times |\mathcal A|)$ のテーブルを出し、各 agent の自分の行動で gather して合計している — これは $\sum_i f_i(s, a^i)$ であり、joint action に条件付いた $f(s,\boldsymbol a)$ ではありません。§3.7 の比較表とラベルを整合させる必要があります。逆に言えば、sum をやめて agent 別に返すだけで F-DAE-agent に移行でき、Phase 2 は目前です。
centering は DAE update が actor update より先に走るので実質 $\mu$ に対して行われており、計画 §5.2 と整合しています。ただし DAE minibatch ごとに確率を再計算しているので、rollout 時に一度キャッシュする方が効率的です。また actor 更新用の advantage を agent 数分だけ重複計算しています（全行が同じ joint 値）。
RMS 正規化（平均を引かない）は centered 構造と credit 比を保つ良い選択です。agent 別 std 正規化は学習した credit 比を壊すので避けるのが正解。
計画 §5.5 は「まず DAE loss + value loss の併用が安全」としていますが、実装は value loss を無効化しています。DAE loss 自体が $V_\phi$ を学習するので二重学習の整合を決めておくべきです（論文は shared network + 係数 $\beta_V$ の一本化）。
SMAC は reward scale が大きく、DAE loss は $r$ と $V$ を同じ二乗の中に入れるため、ValueNorm/PopArt との相互作用（denormalize の位置）を確認してください。
関連研究（差別化必須）
DOP (Wang et al., ICLR 2021)：線形分解 critic による MA policy gradient。F-DAE に最も近い既存研究なので、「Q ではなく advantage を直接推定する」差分を明確に。
HATRPO/HAPPO・MAT：decomposition lemma と順序付き更新。ordered variant を入れるなら中心的参照。
Kuba et al. 2021（MA policy gradient の分散・最適 baseline）：marginal advantage の議論と直結。
COMA が SMAC で MAPPO に大きく負けることは MAPPO 論文で既知なので、「COMA 的 credit assignment を on-policy PPO 系で機能させる」という動機付けに使えます。
まとめ（優先度順の提案）
§4.2 の主張を「COMA advantage の直接推定」から「joint advantage の counterfactual 条件付き centered 分解」へ再定式化する
F-DAE の一意性＋勾配不偏性を命題として形式化する（すぐ書けて Hypothesis 2 を支える）
非一意性の議論を Phase 2 から Phase 3 に移し、CF-DAE の正則化を「最小ノルム分解」として位置付ける。ordered variant を保険として追加する
XOR 一致ゲームを Phase 3 の動機実験として §6 に復活させる
実装のラベル（現状は F-DAE-global）と比較表を整合させる
理論の核（一意性・射影・telescoping）は私のスケッチなので形式証明は必要ですが、どれも数行〜1ページで詰められるはずです。必要なら、この内容を研究計画の md に反映する形で書き加えます。


----


Ordered DAE の詳細 — なぜこれが「一番筋が良い」と考えるか
---
出発点：joint advantage は順序付きで厳密に分解できる
核になるのは HATRPO/HAPPO の multi-agent advantage decomposition lemma です。agent に任意の順序を付けて、

$$
A^i(s, \boldsymbol a^{<i}, a^i) := \mathbb{E}\big[A \mid \boldsymbol a^{\le i}\big] - \mathbb{E}\big[A \mid \boldsymbol a^{<i}\big]
$$

（残りの agent の行動は方策で周辺化）と定義すると、telescoping で和が厳密に $A(s,\boldsymbol a)$ になります。直観的には、$A^i$ は「agent $1..i-1$ の行動が既に決まった状況で、agent $i$ の行動が期待 advantage をどれだけ動かしたか」— 逐次的な限界貢献です。チームの成果を「先に動いた人から順に、その人が加えた差分」として配分するイメージです。

DAE と組み合わせたときの決定的な性質
ここが一番面白いところです。DAE loss は residual に入る和 $\sum_i \hat A_i$ しか見ません。つまり loss 自体は個々の head の値を直接教えてくれない。にもかかわらず：

クラスが完全なので、loss を最小化すれば和は真の $A$ に到達できる（F-DAE との違い）
クラス内で $A$ の表現が一意なので、和が $A$ に一致した瞬間、各 head の値は自動的に $A^i$ に確定する（CF-DAE との違い）
一意性の証明は短いです。$\sum_i f_i = 0$ とし、$\boldsymbol a^{\le k}$ で条件付き期待値を取ると、$i > k$ の項は $f_i$ が自分の行動について centered なので消え、$\sum_{i \le k} f_i = 0$ が任意の $k$ で成立。$k$ を 1 から増やしていけば帰納的に $f_1 = f_2 = \dots = 0$。CF-DAE でこの議論が壊れるのは、$f_i$ が後続の行動 $\boldsymbol a^{>i}$ にも依存できるため、条件付けで消えてくれないからです。「各 head は自分より前の情報にしか依存しない」という causal な構造が、そのまま識別可能性の証明になっている — ここが美しい点です。

つまり O-DAE は「loss は和しか制約しないのに、アーキテクチャの因果構造が credit の帰属を一意に決める」手法であり、CF-DAE が正則化という外付けの装置で解決する問題を、関数クラスの設計だけで解決します。

順序問題と Shapley 分解 — 一番発展性のあるアイデア
弱点は明白で、分解が順序に依存することです。固定順序では agent 1 は常に marginal な（他 agent を平均化した弱い）信号しか受けず、最後の agent は完全に条件付いた強い信号を受けます。

そこで MAT と同様に rollout ごとにランダム順列を使います。このとき起きることが重要で、状態 $s$ ごとに coalition game

$$
v(C) = \mathbb{E}{\boldsymbol a{\bar C} \sim \mu}\big[A(s, \boldsymbol a) \mid \boldsymbol a_C\big]
$$

（「行動が確定した agent 集合 $C$」を提携とみなす）を考えると、順列上で平均した agent $i$ の credit は、まさにこのゲームの Shapley value になります。Shapley value の順列表現そのものだからです。しかも efficiency 公理（$\sum_i \phi_i = v(\text{全体}) = A$）が DAE loss の「和が $A$ を説明する」性質とちょうど噛み合う。つまり：

permutation-averaged O-DAE = 「joint advantage の Shapley 分解」を、Q-function もモンテカルロ提携サンプリングも経由せず、on-policy で直接推定する手法

という位置付けができます。Shapley 系の MARL credit assignment（SQDDPG、Shapley counterfactual credit）は既にありますが、いずれも Q を学習して提携値を近似する構成です。「advantage の分散最小化目的から Shapley credit が副産物として出てくる」という筋は、私の知る限り存在しません（要文献確認 — 後述）。

HAPPO との関係：既存理論への接続と改善可能性
HAPPO は同じ分解を使いますが、$A^i$ を学習せず、importance ratio の積 $M^{1:i} = \prod_{j<i} \frac{\pi_j^{\text{new}}}{\pi_j^{\text{old}}} \cdot \hat A_{\text{joint}}$ でモンテカルロ的に構成します。これは agent の順番が進むほど分散が積で増大する既知の弱点があります。O-DAE はこの $A^i$ を学習された低分散の関数で置き換えるものと見なせるので、

O-DAE 単体：「HAPPO の credit を直接推定する DAE」として理論の借用ができる（monotonic improvement 論法との接続）
逆方向の応用：O-DAE の head を HAPPO の逐次更新に差し込めば、HAPPO 自体の分散問題の改善という副次的な貢献も狙える
論文のストーリーとして、DAE（NeurIPS 2022）と HAPPO（ICLR 2022）という 2 つの確立した理論を橋渡しする形になるのは戦略的にも強いです。

XOR ゲームでの動作イメージ
2 agent 一致ゲーム（一致で +1）、一様方策の対称点で比べると：

F-DAE: $\hat A_i = \mathbb{E}_{a^{-i}}[A] \equiv 0$。学習された決定的な 0 なので勾配もノイズもゼロ — saddle に張り付く
O-DAE(順序 1→2): agent 1 の $A^1$ は marginal で 0 だが、agent 2 の $A^2(a^1, a^2) = \pm 0.5$ は「相手が実際に取った行動に合わせたか」を毎サンプル明確に伝える。ランダム順列なら両 agent が半分の頻度でこの強い信号を受け、SGD ノイズが対称性を破った後は正しい方向に一貫して駆動する
期待値レベルではどの手法も同じ勾配（不偏性）ですが、per-sample の情報量と、対称点を抜ける動力学が違う、というのが正確な主張です。実験ではこの動力学（脱出時間の分布など）を測ると綺麗に差が出るはずです。

実装：現行コードからの差分は小さい
今の実装は centralized features から $N \times |\mathcal{A}|$ の advantage table を出しているので、O-DAE への拡張は：

入力の拡張: 先行 agent の行動の one-hot（後続は zero）＋「誰が先行か」の binary mask を critic 入力に連結。順列は rollout ごとにサンプルして buffer に保存
centering はそのまま: 自分の行動空間の和だけで厳密に計算できる点は現行実装と同一
actor への信号: table から自 agent の行を gather（sum しない）— これは F-DAE-agent への変更と同じ
発展形として、agent 方向に causal mask をかけた attention（MAT decoder と同型）で全 head を 1 パス計算する設計が自然。「advantage decoder」という見た目になり、MAT との対応関係も説明しやすくなります
正直なリスク
学習難度: ランダム順列では head が $2^{N-1}$ 通りの条件付け集合を跨いで汎化する必要があります。SMAC の 5〜10 agent なら現実的ですが、固定順序 → ランダム順列の順で段階的に検証すべきです
命題群は私の導出: 一意性・Shapley 接続・不偏性はスケッチであり、特に DAE の bootstrapped loss（論文 Theorem 2、有限 horizon）での射影幾何は丁寧に詰める必要があります
新規性の確認: 「decomposition lemma の各項を直接推定する」先行研究がないか、投稿前に体系的な文献調査が必要です
進め方の提案
Phase 2（F-DAE-agent、gather 変更だけ）→ XOR/matrix game で F-DAE の信号消失を確認 → O-DAE 固定順で同環境 → ランダム順列 → SMAC、という順序なら、各ステップの実装差分が小さく、理論予測（どこで F-DAE が壊れ、どこで O-DAE が救うか)を一つずつ検証できます。

必要であれば、O-DAE の命題群の形式証明の下書き、matrix game 環境と F-DAE-agent / O-DAE の実装、あるいは Shapley 系・分解系の先行研究の網羅的調査（deep research）から着手できます。