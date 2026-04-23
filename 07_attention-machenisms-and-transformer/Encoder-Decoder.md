## 编码器块：

1. X → 多头自注意力 → Dropout → (+)残差连接（加原始 X）→ LayerNorm → Y
                                                                 │
2. Y → 位置前馈网络 → Dropout → (+)残差连接（加原始 Y）→ LayerNorm → Output

## 解码器块：

1. X → masked多头自注意力 → Dropout → 残差连接（+X）→ LayerNorm → Y

2. Y (作为Query) + 编码器输出 (Key,Value) → 交叉注意力 → Dropout → 残差连接（+Y）→ LayerNorm → Z

3. Z → FFN → Dropout → 残差连接（+Z）→ LayerNorm → 输出