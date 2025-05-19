import graphviz

# 创建一个新的Graphviz Digraph对象
dot = graphviz.Digraph(comment='Model Architecture', format='png')

# 设置节点样式
dot.attr('node', shape='box', style='filled', color='lightgray', fontname='Arial')

# 添加节点
dot.node('input', 'Input\n$x \in \mathbb{R}^6$')
dot.node('gated_classifier', 'Gated Classifier\nRouting Score')
dot.node('joint_optimization', 'Joint Optimization')
dot.node('output', 'Output\n$P(y = 1|x)$')

# 添加专家节点
num_experts = 5  # 假设专家数量为5，可根据实际调整
for i in range(num_experts):
    if i == 0:
        dot.node(f'expert_{i}', 'Expert', shape='oval')
    else:
        dot.node(f'expert_{i}', 'Expert', shape='oval', style='dashed')

# 添加边
dot.edges([
    ('input', 'gated_classifier'),
    ('gated_classifier', 'joint_optimization', label='Update', color='blue'),
    ('joint_optimization', 'gated_classifier', label='Update', color='blue'),
    ('gated_classifier', 'expert_0', label='Data Subset'),
    ('expert_0', 'joint_optimization', label='Feedback Weights'),
    ('joint_optimization', 'expert_0', label='Update', color='blue')
])

# 连接其他专家节点
for i in range(1, num_experts):
    dot.edge('expert_0', f'expert_{i}', style='dashed')

# 连接专家层到输出
dot.edge('joint_optimization', 'output')

# 渲染图形并保存为文件
dot.render('model_architecture', view=True)
