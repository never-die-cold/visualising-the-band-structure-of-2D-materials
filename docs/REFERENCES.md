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
   *Phys. Rev. B* **87**, 125302 (2013).
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
     费米速度 v_F = (√3/2)·a·|t|/ħ ≈ 0.85×10⁶ m/s（t=−2.7 eV, a=2.46 Å）。

## 规划采用（待引入）

6. **Xiao et al. 2012（谷物理/大质量 Dirac 模型原始文献）**
   D. Xiao, G.-B. Liu, W. Feng, X. Xu, W. Yao,
   "Coupled Spin and Valley Physics in Monolayers of MoS₂ and Other Group-VI Dichalcogenides",
   *Phys. Rev. Lett.* **108**, 196802 (2012).
   - 用途：谷选择光学定则、Berry 曲率（ROADMAP：谷物理模块）。

7. **Cudazzo, Tokatly, Rubio 2011 / Keldysh 介电模型（二维屏蔽势）**
   P. Cudazzo, I. V. Tokatly, A. Rubio,
   "Dielectric screening in two-dimensional insulators: Implications for excitonic and
   impurity states in graphane and hexagonal boron nitride",
   *Phys. Rev. B* **84**, 085406 (2011).
   - 用途：ROADMAP 中"介电环境/激子物理"模块的 Rytova-Keldysh 势 V(r)。

8. **Trambly de Laissardière et al. 2010 / Bistritzer-MacDonald 2011（转角石墨烯）**
   G. Trambly de Laissardière, D. Mayou, L. Magaud,
   "Localization of Dirac electrons in rotated graphene bilayers",
   *Nano Lett.* **10**, 804 (2010);
   R. Bistritzer, A. H. MacDonald,
   "Moiré bands and twisted Wannier states",
   *Phys. Rev. B* **84**, 035440 (2011).
   - 用途：ROADMAP 中莫尔超晶格/转角双层引擎（层间 hopping 插值函数）。

## 已删除/待定
（无）
