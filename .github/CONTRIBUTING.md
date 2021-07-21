# Contributing to PennyLane

Thank you for taking the time to contribute to PennyLane!

PennyLane is a collaborative effort with the quantum computation and machine learning communities.
We invite you to join our team in whatever capacity works best for you.

Learn more about contributing to open source in general with
[this great guide](https://opensource.guide/how-to-contribute/).

## How can I get involved in the community?

If you want to contribute but don't know how, start with our
[demonstrations](https://pennylane.ai/qml/demonstrations.html) and
[blog](https://pennylane.ai/blog/). Learn how to do stuff with PennyLane, then begin peeking under
the hood and seeing how that stuff gets done.

To chat directly with the dedicated team behind PennyLane or other members of our community, you
can join our [PennyLane discussion forum](https://discuss.pennylane.ai).

Sometimes, it might take us a couple of hours to reply - please be patient!

## How can I contribute?

It's up to you!

* **Write a Community Demo.** - Show off your PennyLane application on
  [our community page](https://pennylane.ai/qml/demos_community.html). We take Jupyter notebooks,
  scripts with explanations, or entire repositories.  Community demos are a great way to showcase
  research and new papers.

* **Be a part of our community.** - Respond to questions, issues, and
  provide updates on the projects/experiments you are investigating with PennyLane.

* **Test the cutting-edge PennyLane releases.** - Clone our GitHub repository, and keep up with
  the latest features. Learn how to install PennyLane from source
  [here](https://pennylane.ai/install.html?version=preview). If you run into any bugs, make a bug
  report on our [issue tracker](https://github.com/XanaduAI/pennylane/issues).

* **Report bugs.** - If you come across any bugs or issues, make a bug report. See a later section
  for more details on the bug reporting procedure.

* **Suggest new features and enhancements.** Use the GitHub issue tracker and let us know what
  will make PennyLane even better for you.

* **Contribute to the PennyLane repository itself.** See more below.

### Ways to contribute to the PennyLane repository

What to help with the repository itself?  There are several different avenues for that:

- **Good first issues** - Don't know where to start? Take a look at our
  ["good first issue" label](https://github.com/PennyLaneAI/pennylane/contribute).  Issues with
  this label *should* require less expertise and contain fewer tricky bits.  If the issue ends up
  trickier than we assumed, you can start a conversation on the issue page or open up a
  "Work in Progress" PR to ask for help.

- **Documentation**- If you would like to add to our documentation or suggest 
  improvements/changes, let us know or submit a pull request directly. Even Pull Requests fixing
  rendering issues, grammar, or a broken code example can help us. Take a look at the
  [documentation guide](https://pennylane.readthedocs.io/en/stable/development/guide/documentation.html)
  for more specifics.

- **Add a new Template or Operation.** Circuit structures crop up in literature faster than we can
add them, so we are always looking for help. Take a look at the page on
[Contributing templates](https://pennylane.readthedocs.io/en/stable/development/adding_templates.html) 
for more information.

- **Develop a PennyLane plugin** - PennyLane is designed to be device and quantum-framework
  agnostic. Users can switch a circuit's device to any other compatible device with no code
  changes. We would love to support even more devices and quantum frameworks. If you would like to
  write a PennyLane plugin, see the page on
  ["Building a plugin"](https://pennylane.readthedocs.io/en/stable/development/plugins.html).

Did we catch your interest? Let's get into some helpful specifics.

## Details

### Reporting bugs

We use the [GitHub issue tracker](https://github.com/XanaduAI/pennylane/issues) to track all
reported bugs and issues. If you find a bug or have a problem with PennyLane, please submit a bug
report! User reports help us make PennyLane better on all fronts.

To submit a bug report, please consider the following checklist:

* **Search the issue tracker to make sure someone did not already report the bug**. If it was
  already reported, you can add a comment providing more context to those solving the problem.

* **Fill out the issue template**. If you cannot find an existing issue addressing the problem,
  create a new issue by filling out the bug issue form. This template is added automatically to the comment box when you create a new issue. Please try and add as many details as possible!

* Try and make your issue as **clear, concise, and descriptive** as possible. Include an informative title and all code snippets/commands required to reproduce the problem. Try and find the simplest code that reproduces the error you see. If you're not sure what caused the issue,
  describe what you were doing when the issue occurred. Please also include the output of `import pennylane as qml; qml.about()`.

### Suggesting features, document additions, and enhancements

To suggest features and enhancements, please use the GitHub tracker. ADD LINK BEFORE MERGINIG!!!!

* Use a clear and descriptive title.

* Provide a detailed description of the suggested feature.

* List appropriate equations and references.

* Explain how the feature would benefit you and other users.

## Pull requests

If you would like to contribute directly to the PennyLane codebase, make a fork of the master branch and submit a [pull request](https://help.github.com/articles/about-pull-requests). We encourage everyone to fork and modify the PennyLane source code. However, we have a couple of guidelines on pull
requests to ensure the master branch of PennyLane conforms to existing standards and quality. We can help you meet these standards during our code review process, so don't let those stop you from getting started. Pull Requests do not need to be big and complicated.  We appreciate even minor bug fixes or documentation improvements. 

See our [Development Guide](https://pennylane.readthedocs.io/en/stable/development/guide.html)
and the page on [submitting a pull request](https://pennylane.readthedocs.io/en/stable/development/guide/pullrequests.html) in particular for more information.

Thank you for your interest in PennyLane! 🎆
