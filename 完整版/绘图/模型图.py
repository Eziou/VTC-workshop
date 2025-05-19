from graphviz import Digraph

# 创建一个有向图
dot = Digraph(comment='Model Architecture', format='png')
dot.attr(rankdir='TB')  # 从上到下排列

# 定义颜色和样式
dot.attr('node', shape='box', style='filled', fontname='Arial')
dot.attr('edge', fontname='Arial')

# 输入层
dot.node('X', 'Input\nx ∈ ℝⁿ', fillcolor='lightblue')

# 第一层LGBM
dot.node('LGBM1', 'LGBM_1', fillcolor='lightblue')
dot.edge('X', 'LGBM1', label='')

# 第一层输出：K_1 个子集（这里以 K_1=2 为例）
dot.node('S11', 'S_{1,1}', fillcolor='lightgreen')
dot.node('S12', 'S_{1,2}', fillcolor='lightgreen')
dot.edge('LGBM1', 'S11', label='')
dot.edge('LGBM1', 'S12', label='')

# 第二层LGBM（这里以 L=2 为例，即只有两层）
dot.node('LGBM21', 'LGBM_{2,1}', fillcolor='lightblue')
dot.node('LGBM22', 'LGBM_{2,2}', fillcolor='lightblue')
dot.edge('S11', 'LGBM21', label='')
dot.edge('S12', 'LGBM22', label='')

# 第二层输出：概率
dot.node('P1', 'p_{2,1}', fillcolor='lightgreen')
dot.node('P2', 'p_{2,2}', fillcolor='lightgreen')
dot.edge('LGBM21', 'P1', label='')
dot.edge('LGBM22', 'P2', label='')

# 输出层：合并概率
dot.node('P', 'Final Probability p\n(Binary Classification)', fillcolor='lightyellow')
dot.edge('P1', 'P', label='')
dot.edge('P2', 'P', label='')

# 扩展性注释
dot.node('Note', 'Note: Can extend to more subsets (K_l)\nand more layers (L)', fillcolor='white', shape='note')
dot.edge('LGBM22', 'Note', label='', style='dotted')

# 渲染并保存图像
dot.render('model_architecture', view=True)