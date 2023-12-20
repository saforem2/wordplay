# `wordplay` 🎮 💬
Sam Foreman
2023-12-20

<!-- ::: {.quarto-title} -->
<!---->
<!-- ::: {.quarto-title-block} -->
<!---->
<!-- #  [`wordplay` 🎮 💬]{.title} -->
<!---->
<!-- ::: -->
<!---->
<!-- ::: -->

*Playing with words*.

A set of simple, **scalable** and *highly configurable* tools for
working[^1] with LLMs.

## Background

What started as some simple
[modifications](https://github.com/saforem2/nanoGPT) to Andrej
Karpathy's `nanoGPT` has now grown into the `wordplay` project.

<!-- ::: {#fig-compare gap="5%" layout="[[40,40]]" layout-valign="bottom" style="text-align: center!important;" fig-align="center"} -->
<!-- ::: {layout-ncol=2 gap="5%" layout-valign="bottom"} -->
<!-- :::: {.columns layout-ncol=2 layout-valign="bottom" style="margin-bottom: 4em;" style="text-align:center"} -->
<!-- ::: {layout="[15,-10,15]" layout-valign="bottom"} -->

<div class="columns"
style="display: flex; align-items: flex-end; text-align:center; margin-bottom: 2em;">

<div class="column">

<img src="./assets/car.png" style="max-height: 200px" />

</div>

<div class="column">

<img src="./assets/robot.png" style="max-height: 256px" />

</div>

</div>

<details closed>
<summary>
If you’re curious…
</summary>

While `nanoGPT` is a great project and an **excellent** resource; it is,
*by design*, very minimal[^2] and limited in its flexibility.

Working through the code I found myself making minor changes here and
there to test new ideas and run variations on different experiments.
These changes eventually built to the point where *my*
`{goals, scope, code}` for the project had diverged significantly from
the original vision.

As a result, I figured it made more sense to move things to a new
project, [`wordplay`](https://github.com/saforem2/wordplay).

I’ve priortized adding functionality that I have found to be useful or
interesting, but am absolutely open to input or suggestions for
improvement.

Different aspects of this project have been motivated by some of my
recent work on LLMs.

- Projects:
  - [`ezpz`](https://github.com/saforem2/ezpz): Painless distributed
    training with your favorite `{framework, backend}` combo.
  - [`Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed):
    Ongoing research training transformer language models at scale,
    including: BERT & GPT-2
- Collaboration(s):
  - **DeepSpeed4Science** (2023-09)
    - [Loooooooong Sequence Lengths](https://samforeman.me/qmd/dsblog)
    - [Project Website](https://www.deepspeed4science.ai/)
    - [Preprint](https://arxiv.org/abs/2310.04610) Song et al. (2023)
    - [Blog
      Post](https://www.microsoft.com/en-us/research/blog/announcing-the-deepspeed4science-initiative-enabling-large-scale-scientific-discovery-through-sophisticated-ai-system-technologies/)
    - [Tutorial](https://www.deepspeed.ai/deepspeed4science/)
  - GenSLMs:
    - [GitHub](https://github.com/ramanathanlab/genslm)
    - [Preprint](https://www.biorxiv.org/content/10.1101/2022.10.10.511571v2)
    - 🏆 [ACM Gordon Bell Special Prize for COVID-19
      Research](https://www.acm.org/media-center/2022/november/gordon-bell-special-prize-covid-research-2022)
- Talks / Workshops:
  - **LLM-lunch-talk** (2023-10-12): LLMs at
    [ALCF](https://alcf.anl.gov).
    - [Slides](https://saforem2.github.io/llm-lunch-talk/#/section)
    - [GitHub](https://github.com/saforem2/llm-lunch-talk)
  - **Creating Small(-ish) LLMs** (2023-11-30)
    - [Workshop](https://github.com/brettin/llm_tutorial/blob/main/tutorials/03-smallish-LLMs/README.md)
    - [Slides](https://saforem2.github.io/LLM-tutorial/#/creating-small-ish-llmsslides-gh)
    - [GitHub](https://github.com/saforem2/LLM-tutorial)

</details>

## Completed

- [x] Work with *any* 🤗 HuggingFace
  [dataset](https://huggingface.co/docs/datasets/index)
- [x] Effortless distributed training using
  [`ezpz`](https://github.com/saforem2/ezpz)
- [x] Improved (type-safe) and extensible configuration system (powered
  by [`hydra`](https://hydra.cc)), see [\#config](#config)
- [x] Automatic, detailed experiment + metric tracking with [Weights &
  Biases](https://wandb.ai)
  - [Example
    Workspace](https://wandb.ai/l2hmc-qcd/WordPlay?workspace=user-saforem2)
  - [Example
    Run](https://wandb.ai/l2hmc-qcd/WordPlay/runs/in83cm3o/workspace?workspace=user-saforem2)
- [x] [Rich](https://github.com/Textualize/rich) informative logging
  with [`enrich`](https://github.com/saforem2/enrich)

## In Progress

- [ ] [DeepSpeed](https://deepspeed.ai/) support
- [ ] [Full-Sharded Data-Parallel
  (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
  support
  - [Introducing PyTorch Fully Sharded Data Parallel (FSDP) API \|
    PyTorch](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
- [ ] 3D Parallelism support via:
  - [Megatron-DeepSpeed](https://github.com/argonne-lcf/Megatron-DeepSpeed)
  - native PyTorch:
    - [Pipeline Parallelism — PyTorch 2.1
      documentation](https://pytorch.org/docs/stable/pipeline.html)
    - [pytorch/PiPPy: Pipeline Parallelism for
      PyTorch](https://github.com/pytorch/PiPPy)

## Install

<details open>
<summary>
Grab-n-Go
</summary>

The easiest way to get the most recent version is to:

``` bash
python3 -m pip install "git+https://github.com/saforem2/wordplay.git"
```

</details>
<details closed>
<summary>
Development
</summary>

If you’d like to work with the project and run / change things yourself,
I’d recommend installing from a local (editable) clone of this
repository:

``` bash
git clone "https://github.com/saforem2/wordplay"
cd wordplay
mkdir v venv
python3 -m venv venv --system-site-packages
source venv/bin/activate
python3 -m pip install -e .
```

</details>
<!-- # `wordplay` -->
<!---->
<!-- A minimal LLM implementation for research and education. -->
<!-- &title=visitors) -->
<!-- &edge_flat=false) -->
<!-- <p align="center"> -->
<!-- <a href="https://hits.seeyoufarm.com"> -->
<!--     <img align="center" src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fsaforem2.github.io%2Fwordplay&count_bg=%2300CCFF&title_bg=%23303030&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/> -->
<!--   </a> -->
<!-- </p> -->
<!-- ## []{.pink-text} Last Updated -->

------------------------------------------------------------------------

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f; font-style: italic">Last Updated</span>: <span style="color: #f06292; text-decoration-color: #f06292; font-weight: bold">12</span><span style="color: #f06292; text-decoration-color: #f06292">/</span><span style="color: #f06292; text-decoration-color: #f06292; font-weight: bold">20</span><span style="color: #f06292; text-decoration-color: #f06292">/</span><span style="color: #f06292; text-decoration-color: #f06292; font-weight: bold">2023</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">@</span> <span style="color: #1a8fff; text-decoration-color: #1a8fff; font-weight: bold">09:09:58</span>
</pre>

![](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fsaforem2.github.io%2Fwordplay&count_bg=%23222222&title_bg=%23303030&icon=&icon_color=%23E7E7E7)

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0">

<div id="ref-song2023deepspeed4science" class="csl-entry">

Song, Shuaiwen Leon, Bonnie Kruft, Minjia Zhang, Conglong Li, Shiyang
Chen, Chengming Zhang, Masahiro Tanaka, et al. 2023. “DeepSpeed4Science
Initiative: Enabling Large-Scale Scientific Discovery Through
Sophisticated AI System Technologies.”
<https://arxiv.org/abs/2310.04610>.

</div>

</div>

[^1]:

    ``` json
    {
      "training",
      "fine-tuning",
      "benchmarking",
      "parallelizing",
      "distributing",
      "measuring",
      "..."
    }
    ```

    large models at scale.

[^2]: `nano`, even 😂
