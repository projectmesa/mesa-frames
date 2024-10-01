---
title: 'mesa-frames: an extension of mesa for performance and scalability'
tags:
  - Python
  - Mesa
  - Agent-Based Modeling
  - Simulation
  - Visualization
authors:
  - name: Author With ORCID
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: LudwigAuthor with affiliation
    dropping-particle: van
    surname: Surname
    affiliation: 3
affiliations:
 - name: Affiliation 1
   index: 1
   ror: 00hx57361
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).

# Statement of need

`mesa-frames` is an extension to the `mesa` [@python-mesa-2020] agent-based modeling framework in Python, designed to enhance performance and scalability for complex simulations involving millions of agents.

`mesa` has become the most widely used framework for agent-based modeling in Python thanks to its easy-to-use API and object-oriented philosophy. However, iterating through each agent's behavior becomes computationally expensive when the number of agents reaches thousands [@app13010013]. This has led to the development of other frameworks like `Agents.jl` [@Agents.jl], which requires developing in another language, Julia. mesa-frames aims to achieve performance similar to that of `Agents.jl` (NOTE: this needs to be proved) but with a simpler Python syntax.

By storing agents in tabular structures, with agents as rows and attributes as columns, mesa-frames can leverage vectorized operations implemented with speed in mind in lower-level languages. This is achieved through the `Ibis` library, while maintaining a familiar syntax for existing `mesa` users and providing an easy-to-use, expressive API thanks to many out-of-the-box functions implemented in this data manipulation library.

This approach is particularly beneficial for models where agents can "act" simultaneously, a common scenario in fields such as economics, ecology, and social sciences.

Thanks to Ibis being backend-agnostic, even if new, faster DataFrame backends are implemented in the future, changes to the code will be minimal because the API will remain largely unchanged. Additionally, the choice of backends allows very large models to be run in a distributed manner on clusters or using GPUs for acceleration.

The framework's ability to handle large numbers of agents efficiently opens up new possibilities for studying complex systems, from financial markets to epidemiological models, at scales previously challenging with standard `mesa` implementations. This is achieved without requiring efforts to reimplement the code in a lower-level language with more optimizations.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
