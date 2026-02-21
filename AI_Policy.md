# PennyLane AI Tool Use Policy

This document closely follows the [LLVM AI Tool Use Policy](https://github.com/llvm/llvm-project/pull/154441/), with minor adaptations for our ecosystem, and will be applicable across PennyLane, Lightning and Catalyst.

## Policy

PennyLane ecosystem’s policy is that contributors can use whatever tools they would like to  
craft their contributions, but there must be a **human in the loop**. **Contributors must read and review all LLM-generated code or text before they ask other project members to review it.** The contributor is always the author and is fully accountable for their contributions (for further clarification, please see the Copyright section). Contributors should be sufficiently confident that the contribution is high enough quality that asking for a review is a good use of scarce maintainer time, and they should be **able to answer questions about their work** during review.

We expect that new contributors will be less confident in their contributions, and our guidance to them is to \*\*start with small contributions\*\* that they can fully understand to build confidence. We aspire to be a welcoming community that helps new contributors grow their expertise, but learning involves taking small steps, getting feedback, and iterating. Passing maintainer feedback to an LLM doesn't help anyone grow, and does not sustain our community.

Contributors are expected to **be transparent and label contributions that contain substantial amounts of tool-generated content**. Our policy on labelling is intended to facilitate reviews, and not to track which parts of the PennyLane ecosystem are generated. Contributors should note tool usage in their pull request description, commit message, or wherever authorship is normally indicated for the work. For instance, use a commit message trailer like 

```
Assisted-by: <name of code assistant>
```

This transparency helps the community develop best practices and understand the role of these new tools.

This policy includes, but is not limited to, the following kinds of contributions:

- Code, usually in the form of a pull request  
- Issues or security vulnerabilities  
- Comments and feedback on pull requests

## Details

To ensure sufficient self review and understanding of the work, it is strongly recommended that contributors write PR descriptions themselves (if needed, using tools for translation or copy-editing). The description should explain the motivation, implementation approach, expected impact, and any open questions or uncertainties to the same extent as a contribution made without tool assistance.

An important implication of this policy is that it bans agents that take action in our digital spaces without human approval, such as the GitHub [`@claude` agent](https://github.com/claude/). Similarly, automated review tools that publish comments without human review are not allowed. However, an opt-in review tool that **keeps a human in the loop** is acceptable under this policy. As another example, using an LLM to generate documentation, which a contributor manually reviews for correctness, edits, and then posts as a PR, is an approved use of tools under this policy.

AI tools must not be used to fix GitHub issues labelled `good first issue`. These issues are generally not urgent, and are intended to be learning opportunities for new contributors to get familiar with the codebase. Whether you are a newcomer or not, fully automating the process of fixing this issue squanders the learning opportunity and doesn't add much value to the project. **Using AI tools to fix issues labelled as "good first**  
**issues" is forbidden**.

## Extractive Contributions

The reason for our "human-in-the-loop" contribution policy is that processing patches, PRs, RFCs, and comments to open-source software repositories are not free -- it takes a lot of maintainer time and energy to review those contributions! Sending the unreviewed output of an LLM to open source project maintainers ***extracts*** work from them in the form of design and code review, so we call this kind of contribution an "extractive contribution".

Our **golden rule** is that a contribution should be worth more to the project than the time it takes to review it.

## Handling Violations

If a maintainer judges that a contribution doesn't comply with this policy, they should paste the following response to request changes:

```
This PR doesn't appear to comply with our policy on tool-generated content,and requires additional justification for why it is valuable enough to the project for us to review it. 
Please see our developer policy on AI-generated contributions: http://pennylane.ai/docs/AIToolPolicy.html
```

The best way to make a change less extractive and more valuable is to reduce its size or complexity or to increase its usefulness to the community. These factors are impossible to weigh objectively, and our project policy leaves this determination up to the maintainers of the project, i.e., those who are doing the work of sustaining the project.

If or when it becomes clear that a GitHub issue or PR is off-track and not moving in the right direction, maintainers should apply the **`extractive`** label to help other reviewers prioritize their review time.

If a contributor responds but doesn't make their change meaningfully less extractive, maintainers should escalate to the relevant moderation or admin team for the space to lock the conversation.

## Copyright

Artificial intelligence/large-language model systems raise many questions around copyright that have yet to be answered. Our policy on AI/LLM tools is as follows:

Contributors are responsible for ensuring that they have the right to contribute code under the terms of our license, typically meaning that either they, their employer, or their collaborators hold the copyright. Using AI/LLM tools to regenerate copyrighted material does not remove the copyright, and contributors are responsible for ensuring that such material does not appear in their contributions. Contributions found to violate this policy will be removed just like any other offending contribution.

## References

Our policy was informed largely by experience in the LLVM community, with the aforementioned guidelines available at [https://github.com/llvm/llvm-project/pull/154441/](https://github.com/llvm/llvm-project/pull/154441/), which falls under Apache 2.0, and is attributed here.

In addition, the community references the following source for experiences with these tools, and provides justification for the above policies. We follow the LLVM guidelines, and also list the following reference material as source of the above points:

- [Fedora Council Policy Proposal: Policy on AI-Assisted Contributions (fetched 2025-10-01)](https://communityblog.fedoraproject.org/council-policy-proposal-policy-on-ai-assisted-contributions/): Some of the text above was copied from the Fedora  
  project policy proposal, which is licensed under the [Creative Commons  
  Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). This link serves as attribution.
- [Rust draft policy on burdensome PRs](https://github.com/rust-lang/compiler-team/issues/893)
- [Seth Larson's post](https://sethmlarson.dev/slop-security-reports) on slop security reports in the Python ecosystem.
- [Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/).  
- [QEMU bans use of AI content generators](https://www.qemu.org/docs/master/devel/code-provenance.html\#use-of-ai-content-generators)  
- [Slop is the new name for unwanted AI-generated content](https://simonwillison.net/2024/May/8/slop/)
