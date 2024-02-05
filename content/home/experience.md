---
# An instance of the Experience widget.
# Documentation: https://wowchemy.com/docs/page-builder/
widget: experience

# This file represents a page section.
headless: true

# Order that this section appears on the page.
weight: 40

title: Employment
subtitle: Research Experience

# Date format for experience
#   Refer to https://wowchemy.com/docs/customization/#date-format
date_format: Jan 2006

# Experiences.
#   Add/remove as many `experience` items below as you like.
#   Required fields are `title`, `company`, and `date_start`.
#   Leave `date_end` empty if it's your current employer.
#   Begin multi-line descriptions with YAML's `|2-` multi-line prefix.
experience:
  - title: Applied Scientist Intern
    company: Amazon Web Services
    company_url: ''
    company_logo: aws
    location: Cambridge, United Kingdom
    date_start: '2022-06-01'
    date_end: '2022-10-01'
    description: |2-
        As an applied scientist intern at Amazon Web Services (AWS), I led an explorative research project focused on addressing the challenges of hyperparameter optimization for large language models (LLMs). Our primary objective was to gain a comprehensive understanding of the scaling behavior of LLMs and investigate the feasibility of extrapolating optimal hyperparameters from smaller LLMs to their massive counterparts. This hands-on work involved orchestrating the parallel training of multiple LLMs from scratch across numerous GPU cloud instances to gain insights into their scaling dynamics.

        During this internship, I was fortunate to be reunited with [Aaron Klein](https://aaronkl.github.io/), [Matthias Seeger](https://mseeger.github.io/), and [Cédric Archambeau](http://www0.cs.ucl.ac.uk/staff/c.archambeau/), with whom I had previously collaborated during an earlier internship at AWS Berlin.

  - title: Doctoral Placement Researcher
    company: Secondmind
    company_url: https://www.secondmind.ai/
    company_logo: secondmind
    location: Cambridge, United Kingdom
    date_start: '2021-10-01'
    date_end: '2022-05-01'
    description: |2-
        As a student researcher at [Secondmind](https://www.secondmind.ai/) (formerly Prowler.io), a research-intensive AI startup renowned for its innovations in [Bayesian optimization (BO)](tag/bayesian-optimization/) and [Gaussian processes (GPs)](tag/gaussian-processes),  I contributed impactful research and open-source code aligned with their focus on advancing probabilistic ML. Specifically, I developed [open-source software](https://github.com/secondmind-labs/GPflux) to facilitate efficiently sampling from GPs, substantially improving their accessibility and functionality. Additionally, I led a research initiative to improve the integration of neural networks (NNs) with GP approximations, bridging a critical gap between probabilistic methods and deep learning. These efforts culminated in a [research paper](publication/spherical-features-gaussian-process) that was selected for an oral presentation at the prestigious International Conference on Machine Learning (ICML).

        During this period, I had the privilege of working with Vincent Dutordoir and Victor Picheny.

  - title: Applied Scientist Intern
    company: Amazon Web Services
    company_url: ''
    company_logo: aws
    location: Berlin, Germany
    date_start: '2019-06-01'
    date_end: '2019-12-01'
    description: |2-
        As an applied scientist intern at Amazon Web Services (AWS), I contributed to the development of the [Automatic Model Tuning](https://arxiv.org/abs/2012.08489) functionality in [AWS SageMaker](https://aws.amazon.com/sagemaker/). My primary focus was on advancing [AutoML](/tag/automl/) and hyperparameter optimization, particularly [Bayesian optimization (BO)](tag/bayesian-optimization/) methods. I spearheaded a research project aimed at integrating multi-fidelity BO with asynchronous parallelism to significantly improve the efficiency and scalability of model tuning. This initiative led to the development of a [research paper](publication/async-multi-fidelity-hpo/) and the release of open-source code within the [AutoGluon](https://github.com/awslabs/autogluon), subsequently forming the basis of the [SyneTune](https://github.com/awslabs/syne-tune) library.

        I had the privilege of working closely with [Matthias Seeger](https://mseeger.github.io/), [Cédric Archambeau](http://www0.cs.ucl.ac.uk/staff/c.archambeau/), and [Aaron Klein](https://aaronkl.github.io/) during this internship.
    # Collaborating with renowned experts like [Matthias Seeger](https://mseeger.github.io/), [Cédric Archambeau](http://www0.cs.ucl.ac.uk/staff/c.archambeau/), and [Aaron Klein](https://aaronkl.github.io/), I acquired profound knowledge and experience in this domain.
    
    # I spent the Summer-Fall of 2019 at Amazon in Berlin, Germany, conducting research in the area of [AutoML](/tag/automl/) in contribution to the [Automatic Model Tuning](https://arxiv.org/abs/2012.08489) service on [AWS SageMaker](https://aws.amazon.com/sagemaker/). I was fortunate to been given the opportunity to work with eminent researchers in the field, [Matthias Seeger](https://mseeger.github.io/), [Aaron Klein](https://aaronkl.github.io/), and [Cédric Archambeau](http://www0.cs.ucl.ac.uk/staff/c.archambeau/). Together, we tackled the challenges of extending *multi-fidelity Bayesian optimization* with *asynchronous parallelism*. The research developed during my internship culminated in a [research paper](publication/async-multi-fidelity-hpo/) and the release of our [code](https://autogluon.mxnet.io/api/autogluon.searcher.html#gpmultifidelitysearcher) as part of the open-source [AutoGluon](https://github.com/awslabs/autogluon) library.

  - title: Software Engineer
    company: CSIRO's Data61
    company_url: https://data61.csiro.au/
    company_logo: data61
    location: Sydney, Australia
    date_start: '2016-07-01'
    date_end: '2019-04-01'
    description: |2-
        As a machine learning (ML) software engineer at CSIRO's Data61, the AI research division of Australia's national science agency, I was an integral part of the Inference Systems Engineering Team, specializing in probabilistic ML for diverse problem domains. Our focus encompassed areas such as spatial inference and Bayesian experimental design, with a primary emphasis on scalability. I led the development of [new microservices](https://data61.csiro.au/en/Our-Research/Our-Work/Safety-and-Security/Understanding-Risk/Determinant) and contributed to the development of open-source libraries for large-scale [Bayesian deep learning](https://github.com/gradientinstitute/aboleth). I also had a stint with the Graph Analytics Engineering Team, where my contributions to research on graph representation learning led to a [research paper](publication/vi-gcn-2) selected for a spotlight presentation at the [Conference on Neural Information Processing Systems (NeurIPS)](https://proceedings.neurips.cc/paper/2020).

  - title: Software Engineer
    company: National ICT Australia (NICTA)
    company_url: ''
    company_logo: nicta
    location: Sydney, Australia
    date_start: '2015-05-01'
    date_end: '2016-06-30'
    description: |2-
        As a machine learning (ML) software engineer at NICTA, I was part of an interdisciplinary ML research team contributing to the [Big Data Knowledge Discovery](https://research.csiro.au/data61/big-data-knowledge-discovery/) initiative, which engaged with leading scientists across various natural sciences domains to develop Bayesian ML software frameworks to support Australia's evolving scientific research landscape. During this time, I led the development and release of numerous [open-source libraries](https://github.com/NICTA/revrand) for applying Bayesian ML at scale.

  - title: Research Intern
    company: Commonwealth Scientific and Industrial Research Organisation (CSIRO)
    company_url: https://www.csiro.au/
    company_logo: csiro
    location: Sydney, Australia
    date_start: '2013-11-01'
    date_end: '2014-02-01'
    description: |2-
        As a summer vacation scholar at CSIRO's Language and Social Computing team, I applied cutting-edge machine learning (ML) and natural language processing (NLP) techniques to build a robust text classification system for automated sentiment analysis.

# - title: PhD Candidate
#   company: University of Sydney
#   company_url: https://sydney.edu.au/
#   company_logo: usyd
#   location: Sydney, Australia
#   date_start: '2018-01-01'
#   date_end: ''
#   description: ''

design:
  columns: '2'
---
