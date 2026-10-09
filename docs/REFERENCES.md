# 参考文献（vdW Studio 物理模型数据来源）

本文件列出 vdW Studio 中所有结构参数与紧束缚/k·p 模型参数的文献来源。
**原则：代码中出现的每一个物理常数都必须能在下述文献中找到出处，
并通过 `tests/` 中的物理验证测试与文献报告值对账。**

> 防幻觉自检说明：以下论文的 arXiv LaTeX 源码已下载至 `papers/arxiv/`，
> 代码注释中标注的数值均直接摘自这些源文件（非凭记忆）。
> 每个模型都在 `tests/test_engine_*.py` 中与文献给出的基准值（带隙、
> 有效质量、自旋劈裂等）进行数值比对。

## 已采用（arXiv 源码在手）

1. **Kormányos et al. 2015（TMD k·p 模型 + 材料参数总表）**
   A. Kormányos, G. Burkard, M. Gmitra, J. Fabian, V. Zólyomi,
   N. D. Drummond, V. Hatalis,
   "k·p theory for two-dimensional transition metal dichalcogenide semiconductors",
   *2D Materials* **2**, 022001 (2015).
   arXiv: [1410.6666](https://arxiv.org/abs/1410.6666)（源码：`papers/arxiv/kormanyos2015/`）
   - 用途：TMD 晶格常数 a₀ 与硫族间距 d_X-X（Table 1）；
     K 点 k·p 哈密顿量（Eq. 24 及 H_D/H_as/H_3w/H_cub 分解，Eq. 25-29）；
     材料参数 E_bg, γ, α, β, κ, η（Tables 8/9，DFT 与 GW 两套）；
     自旋劈裂 2Δ_vb, 2Δ_cb（Tables 3/4）。

2. **Kormányos et al. 2013（MoS₂ 三带/七带细节、trigonal warping）**
   A. Kormányos, V. Zólyomi, N. D. Drummond, G. Burkard,
   "Monolayer MoS₂: trigonal warping, the Γ valley, and spin-orbit coupling effects",
   *Phys. Rev. B* **88**, 045416 (2013).
   arXiv: [1304.4084](https://arxiv.org/abs/1304.4084)（源码：`papers/arxiv/kormanyos2013/`）

3. **Zahid, Liu, Zhu, Wang, Guo 2013（MoS₂ sp³d⁵ 非正交 Slater-Koster TB）**
   F. Zahid, L. Liu, Y. Zhu, J. Wang, H. Guo,
   "A generic tight-binding model for monolayer, bilayer and bulk MoS₂",
   *AIP Advances* **3**, 052111 (2013), [doi:10.1063/1.4804936](https://doi.org/10.1063/1.4804936).
   arXiv: [1304.0074](https://arxiv.org/abs/1304.0074)（源码：`papers/arxiv/zahid2013/`）
   - 用途：sp³d⁵ Slater-Koster 模型的 96 个参数（Table 3：on-site E_s/E_p/E_d、
     λ_SO、Slater-Koster 能量积分 V_…、重叠积分 S_…）。
   - 验证目标（论文 Table 1/2）：单层 K_v1→K_c 带隙 1.805 eV、
     K_v2→K_c 1.969 eV（价带劈裂 0.164 eV）、
     m_e*(K) ≈ 0.43 m₀、m_h*(K) ≈ 0.46 m₀。

4. **Rudenko & Katsnelson 2014（磷烯四带 TB + 结构）**
   A. N. Rudenko, M. I. Katsnelson,
   "Quasiparticle band structure and tight-binding model for single- and bilayer black phosphorus",
   *Phys. Rev. B* **89**, 201408(R) (2014).
   arXiv: [1404.0618](https://arxiv.org/abs/1404.0618)（源码：`papers/arxiv/rudenko2014/`）
   - 用途：单层黑磷四带 TB 模型 hopping 参数 t₁…t₅（Table 1：
     −1.220 / +3.665 / −0.205 / −0.105 / −0.055 eV 及距离、配位数）。
   - 结构（Brown & Rundqvist 实验结构，同论文采用）：
     链方向 a = 3.3136 Å、褶皱方向 c = 4.3763 Å；
     4 原子/胞，分数坐标 (x, y_pucker) = (0, 0.08056), (½, 0.41944),
     (0, −0.08056), (½, −0.41944)，亚层高度 ±1.0654 Å
     （由 8f (0, 0.10168, 0.08056) 等效位置投影得到，
      与 Table 1 全部 5 个 hopping 距离/配位数自洽）。
   - 验证目标：Γ 点模型带隙 1.52 eV（论文 GW 参考 1.60 eV，模型拟合 ±0.3 eV）。

5. **Reich et al. 2002（石墨烯/hBN 二带 TB）**
   S. Reich, J. Maultzsch, C. Thomsen, P. Ordejón,
   "Tight-binding description of graphene",
   *Phys. Rev. B* **66**, 035412 (2002).
   - 用途：石墨烯/hBN 二带模型 hopping t = −2.7 eV 量级、
     hBN 子格势差。验证目标：石墨烯 K 点 Dirac 锥、
     费米速度 v_F = (√3/2)·a·|t|/ħ ≈ 0.85×10⁶ m/s（t=−2.7 eV, a=2.46 Å）；
     BM moiré 模型的 ħv = (√3/2)|t|a₀ = 5.755 eV·Å 同源。

6. **Xiao, Chang & Niu 2010（Berry 相位波包理论总纲，轨道磁矩）**
   D. Xiao, M.-C. Chang, Q. Niu,
   "Theory of Berry phase effects in materials",
   *Rev. Mod. Phys.* **82**, 1959 (2010).
   arXiv: [0907.2021](https://arxiv.org/abs/0907.2021)（源码：`papers/arxiv/xiao_rmp2010/`）
   - 用途：Berry 曲率（速度算符表述，`analysis/berry.py`）与
     **轨道磁矩**（波包自旋转 Eq. (wave:m)，`analysis/orbital_moment.py`，
     Phase 4.2 谷磁矩/谷 Zeeman）；
   - 关键结果：带边磁矩 m(τ_z) = τ_z·μ*_B，μ*_B = eħ/2m*；
     Zeeman 耦合 ε_M(k) = ε(k) − m(k)·B → 谷劈裂 ΔE = −2m·B。

7. **Berkelbach, Hybertsen & Reichman 2013（TMD 激子参数标定）**
   T. C. Berkelbach, M. S. Hybertsen, D. R. Reichman,
   "Theory of neutral and charged excitons in monolayer transition metal dichalcogenides",
   *Phys. Rev. B* **88**, 045318 (2013).
   arXiv: [1305.4972](https://arxiv.org/abs/1305.4972)（源码：`papers/arxiv/berkelbach2013/`）
   - 用途：MoS₂/MoSe₂/WS₂/WSe₂ 激子默认参数（Table：μ = 0.25/0.27/0.16/0.17 m₀，
     χ₂D = 6.60/8.23/6.03/7.18 Å，变分束缚能 0.54/0.47/0.50/0.45 eV）；
     Keldysh 势形式 V = πe²/((ε₁+ε₂)ρ₀)[H₀−Y₀] 与本模块逐项一致。
   - 验证：精确 DVR 解 ≥ 变分下界且偏差 ≤10%（`tests/test_exciton.py`）。

8. **Bistritzer & MacDonald 2011（转角双层石墨烯连续模型）**
   R. Bistritzer, A. H. MacDonald,
   "Moiré bands in twisted double-layer graphene",
   *PNAS* **108** (30), 12233–12237 (2011), [doi:10.1073/pnas.1108174108](https://doi.org/10.1073/pnas.1108174108).
   arXiv: [1009.4203](https://arxiv.org/abs/1009.4203)（源码：`papers/arxiv/bistritzer2011/`）
   - 用途：`engine/moire.py` 全部模型要素：单层 Dirac 块 h_k(θ)（Eq. 1）、
     层间隧穿 T(r) = wΣe^{−iq_j·r}T_j 与 T₁/T₂/T₃ 矩阵（Eq. 2–5，w=110 meV）、
     q_j 方向 (0,−1)/(√3/2,1/2)/(−√3/2,1/2)、moiré 周期 L_m = a₀/(2sin(θ/2))、
     α = w/(vk_θ) 单参数标度、第一壳层 v*/v = (1−3α²)/(1+6α²)（Eq. 8 与 SI）。
   - 验证（`tests/test_moire.py`）：第一壳层解析速度（≤1%）、
     **魔角 θ=1.05°**（BM Fig. 3）、k=0 双零模（BM SI）、
     魔角平带 ~8 meV vs 2° 时 ~110 meV（BM Fig. 2d）。
   - 实现注记：h 块相位取 θ→0 系综（BM SI 8 带推导同款近似，
     "dependence of h(θ) on angle is parametrically small"），
     该约定保证恒等式 Σ T_j(σ·q̂_j)T_j† = 0 严格成立——
     保留 ±θ/2 相位会以 O(θ) 破坏相消干涉（魔角最小值抬高 ~40 倍）。

9. **Xiao et al. 2012（TMD 谷物理原始文献）**
   D. Xiao, G.-B. Liu, W. Feng, X. Xu, W. Yao,
   "Coupled Spin and Valley Physics in Monolayers of MoS₂ and Other
   Group-VI Dichalcogenides",
   *Phys. Rev. Lett.* **108**, 196802 (2012).
   arXiv: [1112.3144](https://arxiv.org/abs/1112.3144)（源码：`papers/arxiv/xiao2012/`）
   - 用途：大质量 Dirac 模型 Berry 曲率解析式
     Ω_c = −τ·2a²t²Δ′/(Δ′²+4a²t²k²)^{3/2}（`analysis/berry.py` 的
     合成模型对账锚点）；谷选择光学定则与谷 Hall 物理表述；
     Table（a, Δ, t, 2λ, Ω₁, Ω₂）可作 k·p 交叉参考。

## 规划采用（待引入）

9. **Cudazzo, Tokatly & Rubio 2011（二维屏蔽势原始推导）**
   P. Cudazzo, I. V. Tokatly, A. Rubio,
   "Dielectric screening in two-dimensional insulators: Implications for excitonic and
   impurity states in graphane",
   *Phys. Rev. B* **84**, 085406 (2011).
   arXiv: [1104.3346](https://arxiv.org/abs/1104.3346)（源码：`papers/arxiv/cudazzo2011/`）
   - 已用于：r₀ = 2πχ₂D 关系与 Keldysh 势形式的独立交叉确认
     （V = e²/(4α₂D)[H₀−Y₀]，与 Berkelbach Eq. (1) 等价）；
     graphane 激子有效质量参数（α=2.62/m₀, β=0.98/m₀, m_e=0.83m₀）。
   - 注：graphane 的 α₂D 数值未在正文给出（仅图示），
     故未用作 preset 默认值；如需 graphane 激子参数需补文献。

## 已删除/待定
（无）

## Phase 3 进展说明（激子模块，2026-10-07 更新）

2D 激子求解器（Fourier–Bessel DVR + 广义本征值，
`vdw_studio/analysis/exciton.py`）：库仑极限下与 2D 氢原子精确谱
E_n = −4Ry*/(2n+1)² 对账通过（N=250 时误差 <2%）；Keldysh 势的
r₀→0 库仑极限逐点精确。
**材料默认值已标定**：MoS₂/MoSe₂/WS₂/WSe₂ 的 μ 与 χ₂D 取
Berkelbach et al. PRB 88, 045318 (2013) Table（见上文条目 7），
经 `tests/test_exciton.py::TestBerkelbachCalibration` 与文献变分
束缚能对账（精确解 ≥ 变分下界，偏差 2–5%），材料排序
MoS₂ > WS₂ > MoSe₂ > WSe₂ 复现。势形式与 Berkelbach Eq. (1)
及 Cudazzo Eq. (int2) 逐项一致（双重文献交叉确认）。
**仍缺**：hBN / graphane / MoTe₂ / WTe₂ 的 r₀ 数值（需补文献：
hBN 可取 Olsen et al. 或 Latini et al. 的 χ₂D；graphane 需
Cudazzo 正文之外的补充材料）。

## 待确认的约定（欢迎提供文献）

1. **Nanoskif SK 方向/相位约定**（阻塞 Zahid sp³d⁵ 引擎的带隙验证）：
   Zahid 2013 论文只给出 96 个拟合参数值，未写明其 Slater-Koster 表的
   键方向约定与轨道相位约定（Nanoskif 软件内部约定）。当前实现已达成：
   H(k)/S(k) 厄米、S(k) 正定（Gram 判据）、泛函本征值可解；
   但带隙数值与文献目标（单层 K 点 1.805 eV）尚未吻合（当前 ~4.2 eV），
   提示相位约定仍有系统差异。需要以下任一材料确认约定：
   - Zahid F. 的博士论文（香港大学，含 Nanoskif sp³d⁵ 模型细节）；
   - Nanoskif / NEMO-3D 的 sp³d⁵s* 模型文档：
     G. Klimeck et al., *IEEE Trans. Electron Devices* / J. Comput. Electron.
     中 NEMO 3-D 系列论文的 SOC 与 SK 约定附录；
   - 或任何给出 sp³d⁵ Slater-Koster 完整角因子表（含 d-d 交叉项）的
     教材/文献（如 Papaconstantopoulos《Handbook of the Band Structure of
     Elemental Solids》附录）。
   拿到后即可完成该引擎并通过 1.805/1.969 eV 带隙验证。
