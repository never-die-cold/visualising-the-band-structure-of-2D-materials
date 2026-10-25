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
   - 验证：精确 DVR 解 ≥ 变分下界（变分值为基态上界）且偏差 ≤10%，实测偏差 2–5%（`tests/test_exciton.py`；ROADMAP 与检查计划引用的是实测区间，两者不冲突）。

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

10. **Cudazzo, Tokatly & Rubio 2011（二维屏蔽势原始推导）**
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

## sp³d⁵ 引擎状态（2026-10-14 复核，几何/填充已定，SK 约定待定）

**已独立核实并修正两项**（不依赖 Nanoskif 约定）：

1. **几何——三棱柱 vs 八面体（已修正，真实 bug）**：1H 相 MoS₂ 的
   Mo 为**三棱柱配位**（上下 S 三角形对齐、两 S 同一面内空位），
   此前实现把两 S 放在 (1/3,2/3)/(2/3,1/3) 两个不同空位——
   60° 交错排列 = **八面体（1T 型）几何**，晶体场给反。
   核实来源：Wikipedia "Molybdenum disulfide"（"each molybdenum
   atom exists at the center of a trigonal prismatic coordination
   sphere"）+ 2H 体相 P6₃/mmc（Mo 2c / S 4f）坐标的独立距离验算
   （Acta Cryst. B39, 404 (1983) 数据）。已修正
   `structure/builders.py::tmd/tmd_bilayer` 与 `engine/zahid_mos2.py`
   的位点；键长（2.41 Å）与层间 S–S vdW 距离（3.49 Å）锚点均保持。
   **注意：修正改变了 TMD 的 POSCAR/XYZ 导出与结构图（S 方位）**。
2. **中性填充（已确认）**：Mo (4d⁵5s¹) + 2×S (3s²3p⁴) = **18 电子/胞**；
   含 SOC 时 54 态占 18（此前预设假设的"半填 27"错误，已改）。
   Γ 点 27 号态的简并线索即由半填假设引起。

**仍待确认：Nanoskif 的 Slater-Koster 表约定**（阻塞带隙验证）。
现实现下 K 点晶体场顺序反转：带顶为 e′ 型 d_yz/d_zx（正确应为
e 型 d_x²−y²/d_xy，m=±2）；修正几何下 E[18]−E[17] ≈ 0.98 eV
（无 SOC，文献目标 ~1.9，含 SOC 1.805）。

已系统扫描的装配自由度（均不能同时满足"带顶 m=±2 主导 + 带隙
接近目标"）：
- 异种对角色互换（spσ↔psσ 用法）×2；
- 通道族符号翻转（π/δ/奇宇称/pσ/dσ/dd 及其 2⁶ 组合）；
- S–Mo 块单/双通道符号翻转 106 组合。

**关键线索**：π 通道族整体翻转可恢复**正确的带顶字符**
（m=±2 主导；当时的权重数字 0.74 出自错误的投影公式——漏掉子空间
重叠矩阵的逆，2026-10-17 更正，见 ``vdw_studio.fitting.projection``——
π 翻转变体未参数化，修正公式下的数值待重算，定性结论待复核），
但带隙仍偏低（0.76 eV）——指向 π 通道的
双中心坐标系约定，该自由度不能表达为逐通道符号翻转。

**同亚层「± 镜像 + 共轭填充」复核（2026-10-14 负结果）**：
曾怀疑 3 镜像 + 共轭填充约定对奇宇称通道（s-p、p-d 类在
R̂→−R̂ 下变号）有隐含错误，遂试验**同亚层**「全 6 镜像显式列出、
不做共轭填充」（S–Mo 六键本就是显式列出的真实近邻，保持不变）：
结果 H 非厄米（K 点 max|H−H†| ≈ 1.72 eV）且 S(k) 不再厄米。原因：Zahid 表的
E_spσ 与 E_psσ 是**独立拟合参数**（0.6656 vs −1.6515，不是相反数），
因此"物理宇称交换关系"不适用于拟合表——共轭填充（H_bа = H_ab†）
是使模型厄米的必要装配选择，现实现（3 镜像 + 填充）在此意义上正确。
剩余约定差异必在 Nanoskif 的装配/框架约定内部。
该负结果自 2026-10-17 起可由 `ZahidMoS2Model(mirror_images=SIX_MIRROR_IMAGES,
conjugate_fill=False)` 重跑，回归见 `tests/test_zahid_mos2.py::TestKnownIssue`。

需要文献：**Nanoskif 用户参考手册**（NanoAcademic Technologies / 鸿之微科技，未公开发布，须向其索取）。2026-10-17 网络核查的结论：

- 论文正文与 arXiv 源码（`papers/arxiv/zahid2013/`）均未提及方向余弦、
  轨道排序、框架旋转或近邻枚举，且无补充材料。参数由 Nanoskif
  （"Nanostructure Slater-Koster Integrals Fitting package"，NanoAcademic
  开发、鸿之微代理）自动拟合生成；该工具从未公开发布（旧官网自述
  "not yet released"，现产品线只剩 NanoDCAL/RESCU/QTCAD/LatticeMind）。
- Nanoskif 自述"限制在由 Slater 和 Koster 最先提出的传统双中心近似中"，
  并支持正交/非正交基、逐元素轨道基与可配置近邻壳层。这**不支持**
  "非标准旋转轨道框架"的假设。
- 相邻代码的 η 是 Harrison 键长标度指数而非框架旋转（QTCAD：
  V(d) = V(d₀)(d₀/d)^η；NEMO-3D：t(s,p_x) = l'·V_spσ·(d₀/d)^η_spσ）。
  此前记录的"η_ij 型轨道框架旋转"缺少来源，已按此更正。
- NEMO-3D（arXiv:0901.1890）与 QTCAD 用的都是标准 SK54 方向余弦表，
  但二者均为正交 sp³d⁵s* 模型，不能直接证明 2013 非正交表的装配。
- Papaconstantopoulos《Handbook》与其 2003 年 J. Phys. 综述均被付费墙/
  反爬阻断，未能阅读，不构成任何结论。Boykin–Klimeck PRB 69, 115201
  (2004) 同样无法获取。
- 未找到任何独立复现该参数表的后续论文（OpenAlex 全文检索 "Nanoskif" /
  "1.805 eV MoS2 tight-binding" 只命中该论文本身与一篇自拟合参数的
  博士论文）。
- 表自身的结构性事实：S–Mo 块有 8 个独立积分（spσ 0.6656 与 psσ
  −1.6515 等并非相反数），**无法在一次标准 SK54 方向余弦装配中全部
  消费**；论文与可及的 Nanoskif 文档都没说明两种次序如何使用。结合
  Nanoskif 文档化的多壳层（含 E1(S,S)、E1(Mo,Mo)、E2(S,S)）支持，
  最可能的约定错配位置是**逐壳层的方向余弦/参数分配**——这是推断，
  不是已文档化的规则。

**验收门槛（T16）**：拿到约定后对账 K_v1→K_c = 1.805 eV、
K_v2→K_c = 1.969 eV（劈裂 0.164 eV）、m_e* ≈ 0.43 m₀、
m_h* ≈ 0.463 m₀；当前数值在 GUI/CLI/文档中一致标为
"实验性/不得作定量参考"。对账入口：`python examples/sk_convention_scan.py`。
