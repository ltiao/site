---
title: "Probabilistic Machine Learning in the Age of Deep Learning: New Perspectives for Gaussian Processes, Bayesian Optimization and Beyond (PhD Thesis)"
authors:
- admin
date: "2023-09-01T00:00:00Z"
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: "2017-01-01T00:00:00Z"

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ["7"]

# Publication name and optional abbreviated publication name.
# publication: ""
# publication_short: ""

abstract: Advances in artificial intelligence (AI) are rapidly transforming our world, with systems now matching or surpassing human capabilities in areas ranging from game-playing to scientific discovery. Much of this progress traces back to machine learning (ML), particularly deep learning and its ability to uncover meaningful patterns and representations in data. However, true intelligence in AI demands more than raw predictive power; it requires a principled approach to making decisions under uncertainty. This highlights the necessity of probabilistic ML, which offers a systematic framework for reasoning about the unknown through probability theory and Bayesian inference. Gaussian processes (GPs) stand out as a quintessential probabilistic model, offering flexibility, data efficiency, and well-calibrated uncertainty estimates. They are integral to many sequential decision-making algorithms, notably Bayesian optimisation (BO), which has emerged as an indispensable tool for optimising expensive and complex black-box objective functions. While considerable efforts have focused on improving gp scalability, performance gaps persist in practice when compared against neural networks (NNs) due in large to its lack of representation learning capabilities. This, among other natural deficiencies of GPs, has hampered the capacity of BO to address critical real-world optimisation challenges. This thesis aims to unlock the potential of deep learning within probabilistic methods and reciprocally lend probabilistic perspectives to deep learning. The contributions include improving approximations to bridge the gap between GPs and NNs, providing a new formulation of BO that seamlessly accommodates deep learning methods to tackle complex optimisation problems, as well as a probabilistic interpretation of a powerful class of deep generative models for image style transfer. By enriching the interplay between deep learning and probabilistic ML, this thesis advances the foundations of AI and facilitates the development of more capable and dependable automated decision-making systems.


# Summary. An optional shortened abstract.
summary: This thesis explores the intersection of deep learning and probabilistic machine learning to enhance the capabilities of artificial intelligence. It addresses the limitations of Gaussian processes (GPs) in practical applications, particularly in comparison to neural networks (NNs), and proposes advancements such as improved approximations and a novel formulation of Bayesian optimization (BO) that seamlessly integrates deep learning methods. The contributions aim to enrich the interplay between deep learning and probabilistic ML, advancing the foundations of AI and fostering the development of more capable and reliable automated decision-making systems.

tags:
- Machine Learning
- Probabilistic Models
- Gaussian Processes
- Variational Inference
- Bayesian Optimization
- Deep Learning

featured: true

links:
- name: Preprint
  url: phd-thesis-louis-tiao.pdf
- name: Acknowledgements
  url: publication/phd-thesis/gratiae/
# url_pdf: '#'

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  focal_point: Center
  preview_only: true

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
# projects:
# - internal-project

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
# slides: example
---

### Table of Contents

- Chapter 1: Introduction {{< staticref "#" "newtab" >}}{{< icon name="download" pack="fas" >}}{{< /staticref >}}
- Chapter 2: Background {{< staticref "#" "newtab" >}}{{< icon name="download" pack="fas" >}}{{< /staticref >}}
- Chapter 3: Orthogonally-Decoupled Sparse Gaussian Processes with Spherical Neural Network Activation Features {{< staticref "#" "newtab" >}}{{< icon name="download" pack="fas" >}}{{< /staticref >}}
- Chapter 4: Cycle-Consistent Generative Adversarial Networks as a Bayesian Approximation {{< staticref "#" "newtab" >}}{{< icon name="download" pack="fas" >}}{{< /staticref >}}
- Chapter 5: Bayesian Optimisation by Classification with Deep Learning and Beyond {{< staticref "#" "newtab" >}}{{< icon name="download" pack="fas" >}}{{< /staticref >}}
- Chapter 6: Conclusion {{< staticref "#" "newtab" >}}{{< icon name="download" pack="fas" >}}{{< /staticref >}}

## Introduction

<!-- 
{{% callout note %}}
Create your slides in Markdown - click the *Slides* button to check out the example.
{{% /callout %}}

Add the publication's **full text** or **supplementary notes** here. You can use rich formatting such as including [code, math, and images](https://wowchemy.com/docs/content/writing-markdown-latex/).

## Introduction

- {{< staticref "uploads/cv-louis-tiao.pdf" "newtab" >}}{{< icon name="download" pack="fas" >}} Background{{< /staticref >}} 
 -->