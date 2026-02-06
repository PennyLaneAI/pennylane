Deprecations and Removals
=========================

PennyLane is under continuous development and we sometimes need to make breaking changes to improve
the library. When these breaking changes are necessary, we should make sure to give our users time
to update their workflows to adhere to any new implementation before completely removing the old
one. All ongoing and completed deprecations can be found in :doc:`the deprecations page <../deprecations>`.

Deprecating a feature
---------------------

The means of informing users of a breaking change is called a deprecation cycle. Before removing a
feature, for one or more releases of PennyLane, the deprecated feature should advise users of the
upcoming change while preserving its functionality. Here are the steps needed to properly deprecate
a feature:

1. Identify the new or preferred way of achieving the same functionality once the deprecated
   feature is removed. The exception to this is when a feature is being removed for lack of use.

2. At the beginning of the relevant code (e.g., the ``__init__()`` method of a class that's being
   removed), add the following lines of code, filling in the relevant details:

   .. code-block:: python

       warnings.warn(
           "<name-of-feature> is deprecated and will be removed in version <target-version>. "
           "Instead, please use <name-of-preferred-way-to-achieve-functionality>.",
           qp.exceptions.PennyLaneDeprecationWarning,
       )

   If the feature is being relocated, consider rephrasing the warning to discuss relocation as
   opposed to deprecation. As well, if the deprecated feature is fairly minor, it is likely safe to
   state that it will be removed in the next release. Otherwise, it is a good practice to wait 2 or
   more releases before fully removing the feature. Consult the product manager if you are not sure
   how long this should be.

3. If the feature has public-facing docs, include a similar warning message in a visible part of
   its docstring using a ``.. warning::`` Sphinx directive.

4. Replace all uses of the deprecated code within PennyLane's own source code. This is to ensure
   that the warning isn't unexpectedly raised for users who did not personally call the deprecated
   piece of code.

5. Add a test to ensure that the deprecation warning is being raised. It should look similar to the
   following code:

   .. code-block:: python

       def test_my_feature_is_deprecated():
           """Test that my_feature is deprecated."""
           with pytest.warns(qp.exceptions.PennyLaneDeprecationWarning, match="my_feature is deprecated"):
               _ = my_feature()

6. Update any tests that specifically cover the deprecated feature to call it inside a
   ``pytest.warns`` context as shown above. For tests that depend on it but are not written to
   test it specifically, update them to use the new/preferred code.

7. Add an entry to the top of the "Pending deprecations" section of ``doc/development/deprecations.rst``.
   There should be existing examples to follow for style.

8. Add a similar entry to the bottom of the "Deprecations" section in ``doc/releases/changelog-dev.md``.

Additional notes on deprecations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``PennyLaneDeprecationWarning``'s are automatically raised as errors in PennyLane tests. If any
  uses of the deprecated code are accidentally left in (tested) PennyLane code, CI will alert you
  with a failure.
- If a feature is flagged for deprecation but is still (significantly) used within PennyLane, it
  is better to change that in a separate PR before deprecating the feature. You might find that the
  feature should not be deprecated at all!

Removing a deprecated feature
-----------------------------

Once a feature has been deprecated for a sufficiently long time, it is considered safe for removal.
Here are the steps needed to properly remove a deprecated feature:

1. Remove the deprecated source code, along with all tests that cover it.

2. In ``doc/development/deprecations.rst``, move the existing deprecation entry to the "Completed
   deprecation cycles" section below. Be sure to update the language to state that it has been
   removed, and ensure that the PennyLane version in which it's being removed is correct.

3. Add a similar entry to the bottom of the "Breaking Changes" section in ``doc/releases/changelog-dev.md``.
