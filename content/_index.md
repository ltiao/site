---
# Leave the homepage title empty to use the site title
title: ''
date: 2022-10-24
type: landing

sections:
  - block: hero
    content:
      text: |-
        <p style="margin-bottom:300px;"></p>
    design:
      background:
        # Choose a color such as from https://html-color-codes.info
        color: 'black'
        # Text color (true=light, false=dark, or remove for the dynamic theme color).
        text_color_light: true
        video:
          # Name of video in `assets/media/`.
          filename: hero.mp4
          # Post-processing: flip the video horizontally?
          flip: false
  - block: about.biography
    id: about
    content:
      title: Bio
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: admin
  - block: collection
    id: news
    content:
      title: News
      subtitle: ''
      text: ''
      # Choose how many pages you would like to display (0 = all pages)
      count: 5
      # Filter on criteria
      filters:
        folders:
          - post
        author: ""
        category: news
        tag: ""
        exclude_featured: false
        exclude_future: false
        exclude_past: false
        publication_type: ""
      # Choose how many pages you would like to offset by
      offset: 0
      # Page order: descending (desc) or ascending (asc) date.
      order: desc
    design:
      # Choose a layout view
      view: compact
      columns: '2'
  - block: experience
    content:
      title: Employment
      subtitle: Research Experience
      # Date format for experience
      #   Refer to https://docs.hugoblox.com/customization/#date-format
      date_format: Jan 2006
      # Experiences.
      #   Add/remove as many `experience` items below as you like.
      #   Required fields are `title`, `company`, and `date_start`.
      #   Leave `date_end` empty if it's your current employer.
      #   Begin multi-line descriptions with YAML's `|2-` multi-line prefix.
      items:
        # - title: Research Scientist
        #   company: Meta Platforms
        #   company_url: ''
        #   company_logo: meta
        #   location: New York City, USA
        #   date_start: '2024-06-17'
        #   date_end: ''
        #   description: 'TBD'
        # - title: Research Associate
        #   company: University of Oxford
        #   company_url: ''
        #   company_logo: oxford
        #   location: Oxford, United Kingdom
        #   date_start: '2024-05-20'
        #   date_end: ''
        #   description: 'TBD'
        - title: Applied Scientist Intern
          company: Amazon Web Services
          company_url: ''
          company_logo: aws
          location: Cambridge, United Kingdom
          date_start: '2022-05-30'
          date_end: '2022-09-16'
          description: |2-
              As an applied scientist intern at Amazon Web Services (AWS), I led an explorative research project focused on addressing the challenges of hyperparameter optimization for large language models (LLMs). Our primary objective was to gain a comprehensive understanding of the scaling behavior of LLMs and investigate the feasibility of extrapolating optimal hyperparameters from smaller LLMs to their massive counterparts. This hands-on work involved orchestrating the parallel training of multiple LLMs from scratch across numerous GPU cloud instances to gain insights into their scaling dynamics.

              During this internship, I was fortunate to be reunited with [Aaron Klein](https://aaronkl.github.io/), [Matthias Seeger](https://mseeger.github.io/), and [Cédric Archambeau](http://www0.cs.ucl.ac.uk/staff/c.archambeau/), with whom I had previously collaborated during an earlier internship at AWS Berlin.
        - title: Doctoral Placement Researcher
          company: Secondmind
          company_url: https://www.secondmind.ai/
          company_logo: secondmind
          location: Cambridge, United Kingdom
          date_start: '2021-09-20'
          date_end: '2022-04-30'
          description: |2-
              As a student researcher at [Secondmind](https://www.secondmind.ai/) (formerly Prowler.io), a research-intensive AI startup renowned for its innovations in [Bayesian optimization (BO)](tag/bayesian-optimization/) and [Gaussian processes (GPs)](tag/gaussian-processes), I contributed impactful research and open-source code aligned with their focus on advancing probabilistic ML. Specifically, I developed [open-source software](https://github.com/secondmind-labs/GPflux) to facilitate sampling efficiently from GPs, substantially improving their accessibility and functionality. Additionally, I led a research initiative to improve the integration of neural networks (NNs) with GP approximations, bridging a critical gap between probabilistic methods and deep learning. These efforts culminated in a [research paper](publication/spherical-features-gaussian-process) that was selected for an oral presentation at the [International Conference on Machine Learning (ICML)](#).

              I had the privilege of collaborating closely with Vincent Dutordoir and Victor Picheny during this period.
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
    design:
      columns: '2'
  - block: collection
    id: featured
    content:
      title: Select Publications
      filters:
        folders:
          - publication
        featured_only: true
    design:
      columns: '2'
      view: card
  - block: collection
    id: posts
    content:
      title: Recent Posts
      subtitle: ''
      text: ''
      # Choose how many pages you would like to display (0 = all pages)
      count: 8
      # Filter on criteria
      filters:
        folders:
          - post
        author: ""
        category: technical
        tag: ""
        exclude_featured: false
        exclude_future: false
        exclude_past: false
        publication_type: ""
      # Choose how many pages you would like to offset by
      offset: 0
      # Page order: descending (desc) or ascending (asc) date.
      order: desc
    design:
      # Choose a layout view
      view: card
      columns: '2'
  - block: portfolio
    id: projects
    content:
      title: Projects
      filters:
        folders:
          - project
      # Default filter index (e.g. 0 corresponds to the first `filter_button` instance below).
      default_button_index: 1
      # Filter toolbar (optional).
      # Add or remove as many filters (`filter_button` instances) as you like.
      # To show all items, set `tag` to "*".
      # To filter by a specific tag, set `tag` to an existing tag name.
      # To remove the toolbar, delete the entire `filter_button` block.
      buttons:
        - name: All
          tag: '*'
        - name: Animations
          tag: Animations
    design:
      # Choose how many columns the section has. Valid values: '1' or '2'.
      columns: '2'
      view: card
      # For Showcase view, flip alternate rows?
      flip_alt_rows: false
  - block: collection
    id: talks
    content:
      title: Recent & Upcoming Talks
      filters:
        folders:
          - event
    design:
      columns: '2'
      view: compact
  - block: markdown
    id: teaching
    content:
      title: Teaching
      subtitle: Courses
      text: |2-
        ## COMP9418: Advanced Topics in Statistical Machine Learning (UNSW Sydney)

        This course has a primary focus on probabilistic machine learning methods, covering the topics of exact and approximate inference in directed and undirected probabilistic graphical models -- continuous latent variable models, structured prediction models, and non-parametric models based on Gaussian processes. 

        {{< figure src="cs9418-lab-screenshots/gp-regression-title.png" caption="Lab exercise on Gaussian Process Regression, running in JupyterLab." >}}

        This course emphasized maintaining a good balance between theory and practice. As the teaching assistant (TA) for this course, my primary responsibility was to create lab exercises that aid students in gaining hands-on experience with these methods, specifically applying them to real-world data using the most current tools and libraries. The labs were Python-based, and relied heavily on the Python scientific computing and data analysis stack ([NumPy], [SciPy], [Matplotlib], [Seaborn], [Pandas], [IPython/Jupyter notebooks]), and the popular machine learning libraries [scikit-learn] and [TensorFlow].

        Students were given the chance to experiment with a broad range of methods on various problems, such as Markov chain Monte Carlo (MCMC) for Bayesian logistic regression, probabilistic PCA (PPCA), factor analysis (FA) and independent component analysis (ICA) for dimensionality reduction, hidden Markov models (HMMs) for speech recognition, conditional random fields (CRFs) for named-entity recognition, and Gaussian processes (GPs) for regression and classification.

        [NumPy]: http://www.numpy.org/
        [SciPy]: https://www.scipy.org/
        [Matplotlib]: https://matplotlib.org/
        [Seaborn]: https://seaborn.pydata.org/
        [Pandas]: https://pandas.pydata.org/
        [IPython/Jupyter notebooks]: http://jupyter.org/
        [scikit-learn]: http://scikit-learn.org/
        [TensorFlow]: https://www.tensorflow.org/
        [JupyterLab]: https://blog.jupyter.org/jupyterlab-is-ready-for-users-5a6f039b8906
        [COMP9418]: http://www.handbook.unsw.edu.au/postgraduate/courses/2017/COMP9418.html
        [UNSW Sydney]: https://www.unsw.edu.au/
    design:
      columns: '2'
  - block: collection
    content:
      title: Publications
      text: |-
        {{% callout note %}}
        Quickly discover relevant content by [filtering publications](./publication/).
        {{% /callout %}}
      filters:
        folders:
          - publication
        exclude_featured: false
    design:
      columns: '2'
      view: citation
  - block: tag_cloud
    content:
      title: Topics
    design:
      columns: '2'
  - block: contact
    id: contact
    content:
      title: Contact
      subtitle: Get in touch
      text: |-
        Leave me a message:
      # Contact (add or remove contact options as necessary)
      email: louis@tiao.io
      appointment_url: 'https://calendly.com/louistiao/30min'
      contact_links:
        - icon: twitter
          icon_pack: fab
          name: DM me on Twitter
          link: 'https://twitter.com/louistiao'
      # Automatically link email and phone or display as text?
      autolink: true
      # Email form provider
      form:
        provider: netlify
        formspree:
          id:
        netlify:
          # Enable CAPTCHA challenge to reduce spam?
          captcha: true
    design:
      columns: '2'
---
