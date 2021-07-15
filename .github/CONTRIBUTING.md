# Contributing to PennyLane

Thank you for taking the time to contribute to PennyLane!
:confetti_ball: :tada: :fireworks: :balloon:

PennyLane is a collaborative effort with the quantum computation and machine learning
community - while we will continue working on adding new and exciting features to PennyLane,
we invite you to join us and suggest improvements, research ideas, or even just to discuss how
PennyLane fits into your workflow.

## How can I get involved in the community?

If you want to contribute but don't know where to start, start by checking out our
[documentation](https://pennylane.readthedocs.io). Have a go working through some of the tutorials,
and having a look at the PennyLane API and code documentation to see how things work under the hood.

To chat directly with the team designing and building PennyLane, as well as members of our community — ranging from quantum machine learning researchers, to students, to those just interested in being a part of a rapidly growing industry — you can join our [PennyLane discussion forum](https://discuss.pennylane.ai).

Available categories include:

* `PennyLane Feedback`: For general discussion regarding PennyLane, including feature requests, and theoretical questions
* `PennyLane Help`: For help and advice using PennyLane
* `PennyLane Development`: For discussion of PennyLane development
* `PennyLane Plugins`: For discussion of the available PennyLane plugins, and plugin development
* `Xanadu Software`: For discussion relating to other Xanadu software projects, including [Strawberry Fields](https://github.com/xanaduai/strawberryfields) and [The Walrus](https://github.com/xanaduai/thewalrus).

Sometimes, it might take us a couple of hours to reply - please be patient!

## How can I contribute?

It's up to you!

* **Community Demos** - Show off your PennyLane application on [our community page](https://pennylane.ai/qml/demos_community.html). We take Jupyter notebooks, scripts with explanations, or full repositories.  Community demos are a great way to showcase research and new paper.

* **Be a part of our community** - respond to questions, issues, and
  provide exciting updates of the projects/experiments you are investigating with PennyLane

  You can even write your own PennyLane tutorials, or blog about your results.
  Send us the link, and we may even add it to our documentation as an external resource!

* **Test the cutting-edge PennyLane releases** - clone our GitHub repository, and keep up to
  date with the latest features. If you run into any bugs, make a bug report on our
  [issue tracker](https://github.com/XanaduAI/pennylane/issues).

* **Report bugs** - even if you are not using the development branch of PennyLane, if you come
  across any bugs or issues, make a bug report. See the next section for more details on the bug
  reporting procedure.

* **Suggest new features and enhancements** - use the GitHub issue tracker
  and let us know what will make PennyLane even better for your workflow.

* **Contribute to our documentation, or to PennyLane directly** - if you would like to add
  to our documentation, or suggest improvements/changes, let us know - or even submit a pull request directly. All authors
  who have contributed to the PennyLane codebase will be listed alongside the latest release. Even Pull Requests fixing
  rendering issues, grammar, or a code example that doesn't work as expected can really help us out. Take a look at
  the [documentation page](https://pennylane.readthedocs.io/en/stable/development/guide/documentation.html) for more specifics.


* **Add a new Template or Operation** - Circuit structures crop up in literature faster than we can add them, so we are 
  always looking for help. Take a look at the page on 
  [Contributing templates](https://pennylane.readthedocs.io/en/stable/development/adding_templates.html)
  for more information.

* **Develop a PennyLane plugin** - PennyLane is designed to be device and quantum-framework agnostic;
  the quantum node device can be switched out to any other compatible devices, with no code changes necessary.
  We would love to support even more devices and quantum frameworks. If you would like to write a PennyLane plugin,
  see the page on ["Building a plugin"](https://pennylane.readthedocs.io/en/stable/development/plugins.html).

  Ask us if you have any questions, and send a link to your plugin to support@xanadu.ai so we can highlight it in
  our documentation!

Appetite whetted? Let's get into some helpful specifics.

## Details

### Reporting bugs

We use the [GitHub issue tracker](https://github.com/XanaduAI/pennylane/issues) to keep track of all reported
bugs and issues. If you find a bug, or have an issue with PennyLane, please submit a bug report! User
reports help us make PennyLane better on all fronts.

To submit a bug report, please work your way through the following checklist:

* **Search the issue tracker to make sure the bug wasn't previously reported**. If it was, you can add a comment
  to expand on the issue if you would like.

* **Fill out the issue template**. If you cannot find an existing issue addressing the problem, create a new
  issue by filling out the [issue template](ISSUE_TEMPLATE.md). This template is added automatically to the comment
  box when you create a new issue. Please try and add as many details as possible!

* Try and make your issue as **clear, concise, and descriptive** as possible. Include a clear and descriptive title,
  and include all code snippets/commands required to reproduce the problem. Try and find the simplest code that
  reproduces the problem you are seeing. If you're not sure what caused the issue,
  describe what you were doing when the issue occurred. Please also include the output of `import pennylane as qml; qml.about()`.

### Suggesting features, document additions, and enhancements

To suggest features and enhancements, please use the GitHub tracker. There is no template required for
feature requests and enhancements, but here are a couple of suggestions for things to include.

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested feature**.

  - If the feature is related to any theoretical results in quantum machine learning or quantum computation,
    provide any relevant equations. Alternatively, provide references to papers/preprints,
    with the relevant sections/equations noted.
  - If the feature is workflow-related, or related to the use of PennyLane,
    explain why the enhancement would be useful, and where/how you would like to use it.

* **For documentation additions**, point us towards any relevant equations/papers/preprints,
    with the relevant sections/equations noted. Short descriptions of its use/importance would also be useful.

## Pull requests

If you would like to contribute directly to the PennyLane codebase, simply make a fork of the master branch, and
then when ready, submit a [pull request](https://help.github.com/articles/about-pull-requests). We encourage everyone
to have a go forking and modifying the PennyLane source code, however, we have a couple of guidelines on pull
requests to ensure the main master branch of PennyLane conforms to existing standards and quality.

For more information, see our [Development Guide](https://pennylane.readthedocs.io/en/stable/development/guide.html),
and the page on [submitting a pull request](https://pennylane.readthedocs.io/en/stable/development/guide/pullrequests.html) in particular.

:fireworks: Thank you for contributing to PennyLane! :fireworks:

\- The PennyLane team
