<h1 align="center">
  <a href="https://github.com/luchris429/ai_scientist/blob/main/docs/logo_2.png">
    <img src="docs/logo_2.png" width="215" /></a><br>
  <b>The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery 🧑‍🔬</b><br>
</h1>

<p align="center">
  📚 <a href="https://arxiv.org/abs/your_paper_placeholder">[Paper]</a> |
  📝 <a href="https://sakana.ai/ai-scientist/">[Blog Post]</a> |
  📂 <a href="https://drive.google.com/drive/folders/1vraDwQV_xVD4r8xfj8NTn5B0ncOdfkR-">[Drive Folder]</a>
</p>

One of the grand challenges of artificial intelligence is developing agents capable of conducting scientific research and discovering new knowledge. While frontier models have already been used to aid human scientists, e.g. for brainstorming ideas or writing code, they still require extensive manual supervision or are heavily constrained to a specific task.

We're excited to introduce The AI Scientist, the first comprehensive system for fully automatic scientific discovery, enabling Foundation Models such as Large Language Models (LLMs) to perform research independently.

We further provide all runs and data from our paper [here](https://drive.google.com/drive/folders/1G7A0wTqfXVa-cpexjk0oaXakaSJwffEt?usp=sharing), where we run each base model on each template for ~50 ideas. We *highly* recommend reading through some of the [claude papers](https://drive.google.com/drive/folders/1Mmpz6M1FK4q8e-SewgZcUzdeD0Q2zC39?usp=sharing), (especially the diffusion ones), to get a sense of its strengths and weaknesses.

**Note**: Caution! This codebase will execute LLM-written code. There are various risks and challenges associated with this autonomy. This includes e.g. the use of potentially dangerous packages, web access, and potential spawning of processes. Use at your own discretion. Please make sure to containerize and restrict web access appropriately.

<p align="center">
  <img src="docs/adaptive_dual_scale_denoising.jpeg" alt="Adaptive Dual Scale Denoising" width="80%" />
</p>

## Requirements

### Installation

```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# LLM APIs
pip install anthropic aider-chat backoff openai
# Viz
pip install matplotlib pypdf pymupdf4llm
# Install pdflatex
sudo apt-get install texlive-full

# Common Requirements
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

We use the following environment variables for the different API providers for different models:

`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`

Our code can also optionally use a Semantic Scholar API Key (`S2_API_KEY`) for higher throughput [if you have one](https://www.semanticscholar.org/product/api), though in principle it should work without it.

Be sure to provide the key for the model used for your runs, e.g.

```
export OPENAI_API_KEY="YOUR KEY HERE"
export S2_API_KEY="YOUR KEY HERE"
```

### Setup NanoGPT

```bash
# Prepare NanoGPT data
python data/enwik8/prepare.py
python data/shakespeare_char/prepare.py
python data/text8/prepare.py
```

#### Create baseline runs (machine dependent)

```
# Set up NanoGPT baseline run
python templates/nanoGPT/experiment.py --out_dir run_0

# Make sure plotting works
python templates/nanoGPT/plot.py
```

#### Create NanoGPT_lite baseline run. We use this for sanity-checking
```
python templates/nanoGPT_lite/experiment.py --out_dir run_0
```

### Setup 2D Diffusion

```bash
# Set up 2D Diffusion
git clone https://github.com/gregversteeg/NPEET.git
cd NPEET
pip install .
pip install scikit-learn

# Set up 2D Diffusion baseline run
python templates/2d_diffusion/experiment.py --out_dir run_0
python templates/2d_diffusion/plot.py
```

### Setup Grokking

```bash
# Set up Grokking baseline run
python templates/grokking/experiment.py --out_dir run_0
python templates/grokking/plot.py
```


## Run AI Scientist Paper Generation Experiments

```bash
conda activate ai_scientist
# Run the paper generation.
python launch_scientist.py --model "gpt-4o-2024-05-13" --experiment nanoGPT_lite --num-ideas 2
python launch_scientist.py --model "claude-3-5-sonnet-20240620" --experiment nanoGPT_lite --num-ideas 2
```

## Getting an LLM Generated Paper Review

```python
import openai
from ai_scientist.perform_review import load_paper, get_llm_review

client = openai.OpenAI()
model = "gpt-4o-2024-05-13"

# Load paper from pdf file (raw text)
paper_txt = load_paper("report.pdf")
# Get the review dict of the review
review = get_llm_review(
    paper_txt,
    model,
    client,
    num_reflections=5,
    num_fs_examples=1,
    num_reviews_ensemble=5,
    temperature=0.1,
)

# Inspect review results
review["Overall"]  # overall score 1-10
review["Decision"]  # ['Accept', 'Reject']
review["Weaknesses"]  # List of weaknesses (str)
```

To run batch analysis:

```bash
cd review_iclr_bench
python iclr_analysis.py --num_reviews 500  --batch_size 100 --num_fs_examples 1 --num_reflections 5 --temperature 0.1 --num_reviews_ensemble 5
```

## Making your own Template

If there is an area of study you would like the AI Scientist to explore, it should be very easy to create your own templates. In general, follow the structure of the existing templates, which consists of:

- `experiment.py` -- This is a single file where the 'meat' of the content is. It takes in an argument for `out_dir`, which is where it should create the folder and save the relevant information from the run.
- `plot.py` -- This should take in the information from the `run` folders and create plots. The code should be clear and easy to edit.
- `prompt.json` -- Put information about your template here.
- `seed_ideas.json` -- Put example ideas here. You can also try to generate ideas without any examples, and then pick the best one or two to put here.
- `latex/template.tex` -- We recommend using our latex folder, but be sure to replace the pre-loaded citations with ones that you would expect to be more relevant.
   
## Template Resources

We provide 3 templates, which heavily use code from other repositories, which we credit below. (Normally, we would do this in the files themselves, but it's unclear how this would affect The AI Scientist since it would be visible).

The NanoGPT template used code from [NanoGPT](https://github.com/karpathy/nanoGPT) and this [PR](https://github.com/karpathy/nanoGPT/pull/254).

The 2D Diffusion template used code from [tiny-diffusion](https://github.com/tanelp/tiny-diffusion) and [ema-pytorch](https://github.com/lucidrains/ema-pytorch).

The Grokking template used code from [Sea-Snell/grokking](https://github.com/Sea-Snell/grokking) and [danielmamay/grokking](https://github.com/danielmamay/grokking).

## Citing `The AI-Scientist` ✏️

If you use `The AI-Scientist` in your research, please cite it as follows:

```
@article{lu2024aiscientist,
  title={The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery},
  author={Lu, Chris and Lu, Cong and Lange, Robert and Foerster, Jakob N and Clune, Jeff and Ha, David},
  journal={arXiv preprint},
  year={2024}
}
```

We would like to thank the developers of the open-source models and packages for their contributions and for making their work available.
