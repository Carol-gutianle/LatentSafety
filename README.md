# LatentSafety

This repository contains the official implementation of the paper  
**"Probing the Safety Robustness of LLMs in Latent Space"**.


## Installation

```bash
git clone {Url to Repo}
cd LatentSafety
pip install -r requirements.txt
```


## ASA Attack

```bash
python -m exp.effectiveness \
    --model_name_or_path DeepSeek-R1-Distill-Qwen-1.5B \
    --seed 42 \
    --jailbreak random \
    --dataset advbench \
    --max_new_tokens 50 \
    --num_samples 100
```

Output file format:

```text
{seed}_{jailbreak}_{max_new_tokens}_{model_name}_{num_samples}.json
```

- `--jailbreak gasa`: enables gradient-based variant (ASA_grad)
- `--jailbreak trojan`: uses activation difference-based perturbations


## Annotation

```bash
python -m exp.annotate \
    --input_file {input}.json \
    --output_file {input}_annotated.json \
    --target steered_response
```

Aggregate results and compute MASR / PASR:

```bash
python -m exp.aggregate
```

Example output (`summary.json`):

```json
{
  "model": "Qwen2.5-7B",
  "total_prompts": 100,
  "successful_prompts": 85,
  "MASR": 0.85,
  "PASR": 0.63
}
```

## ASABench

We release **ASABench**, a benchmark for evaluating safety robustness under latent-space perturbations.

The dataset is available on HuggingFace:

```text
https://huggingface.co/datasets/Carol0110/ASABench
```

Each sample corresponds to a layer-wise probing instance, including:
- prompt
- original response
- perturbed (ASA) response
- layer index
- attack success annotation

The dataset is organized by model (as subsets), enabling comparison across architectures.

ASABench complements behavior-level safety benchmarks by providing a latent-space evaluation protocol.


## Generating ASABench

```bash
python -m exp.asa \
    --model_name_or_path {model path} \
    --model_name {model_name} \
    --save_path {output_path} \
    --mode lapt
```

## Layer-wise Adversarial Patch Training (LAPT)

```bash
python -m train
```

## Visualization Tools

### NLL Curve (ASA / ASA_grad)

```markdown
![](./assets/loss_asa.png)
```

```bash
python -m sv.loss
```

### Loss Landscape

```markdown
![](./assets/landscape.png)
```

```bash
python -m exp.landscape
```

### Cite Us

```bash
@article{gu2025probing,
  title={Probing the robustness of large language models safety to latent perturbations},
  author={Gu, Tianle and Huang, Kexin and Wang, Zongqi and Wang, Yixu and Li, Jie and Yao, Yuanqi and Yao, Yang and Yang, Yujiu and Teng, Yan and Wang, Yingchun},
  journal={arXiv preprint arXiv:2506.16078},
  year={2025}
}
```
