# Model Introduction

```mermaid
classDiagram

    class Bottleneck {
        +expansion: int = 4
        +__init__(inplanes, planes, stride)
        +forward(x)
    }

    class AttentionPool2d {
        +__init__(spacial_dim, embed_dim, num_heads, output_dim)
        +forward(x)
    }

    class ModifiedResNet {
        +__init__(layers, output_dim, heads, input_resolution, width)
        +forward(x)
    }

    class LayerNorm {
        +forward(x)
    }

    class QuickGELU {
        +forward(x)
    }

    class ResidualAttentionBlock {
        +__init__(d_model, n_head, attn_mask)
        +forward(x)
    }

    class Transformer {
        +__init__(width, layers, heads, attn_mask)
        +forward(x)
    }

    class VisionTransformer {
        +__init__(input_resolution, patch_size, width, layers, heads, output_dim)
        +forward(x)
    }

    class CLIP {
        +__init__(...)
        +encode_image(image)
        +encode_text(text)
        +forward(image, text)
        .. visual: ModifiedResNet or VisionTransformer ..
    }

    %% 组合关系（仅自定义类之间）

    ModifiedResNet *-- "1..*" Bottleneck : layers
    ModifiedResNet *-- "1" AttentionPool2d : attnpool

    VisionTransformer *-- "1" Transformer : transformer
    VisionTransformer *-- "2" LayerNorm : ln_pre, ln_post

    Transformer *-- "1..*" ResidualAttentionBlock : resblocks

    ResidualAttentionBlock *-- "2" LayerNorm : ln_1, ln_2
    ResidualAttentionBlock *-- "1" QuickGELU : in mlp
    CLIP *-- "1" ModifiedResNet : visual (ResNet)
    CLIP *-- "1" VisionTransformer : visual (ViT)
    CLIP *-- "1" Transformer : text transformer
    CLIP *-- "1" LayerNorm : ln_final
```
