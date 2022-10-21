---
title: "Model-based Asynchronous Hyperparameter and Neural Architecture Search"
authors:
- Aaron Klein
- admin
- Thibaut Lienart
- Cédric Archambeau
- Matthias Seeger
date: "2020-03-01T00:00:00Z"
doi: ""

# Author notes (optional)
author_notes:
  - 'Equal contribution'
  - 'Equal contribution'

# Schedule page publish date (NOT publication's date).
# publishDate: "2017-01-01T00:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["3"]

# Publication name and optional abbreviated publication name.
publication: ""
publication_short: ""

abstract: We introduce a model-based asynchronous multi-fidelity method for hyperparameter and neural architecture search that combines the strengths of asynchronous Hyperband and Gaussian process-based Bayesian optimization. At the heart of our method is a probabilistic model that can simultaneously reason across hyperparameters and resource levels, and supports decision-making in the presence of pending evaluations. We demonstrate the effectiveness of our method on a wide range of challenging benchmarks, for tabular data, image classification and language modelling, and report substantial speed-ups over current state-of-the-art methods. Our new methods, along with asynchronous baselines, are implemented in a distributed framework which will be open sourced along with this publication.

# Summary. An optional shortened abstract.
# summary: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis posuere tellus ac convallis placerat. Proin tincidunt magna sed ex sollicitudin condimentum.

tags:
- Machine Learning
- AutoML
- Bayesian Optimization
- Hyperparameter Optimization
- Gaussian Processes
- Parallel Computing
featured: true

links:
url_pdf: https://arxiv.org/abs/2003.10865
url_code: https://autogluon.mxnet.io/api/autogluon.searcher.html#gpmultifidelitysearcher
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: https://www.youtube.com/watch?v=sBFICoq2pbg

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: Behavior of synchronous BOHB compared to MoBSter, an asynchronous extension of BOHB based on Gaussian processes.
  focal_point: Center
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: []

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""
---
