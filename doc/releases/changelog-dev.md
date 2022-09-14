:orphan:

# Release 0.27.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

* Structural improvements are made to `QueuingContext`, now `QueuingManager`, and `AnnotatedQueue`.
  [(#2794)](https://github.com/PennyLaneAI/pennylane/pull/2794)
  [(#3061)](https://github.com/PennyLaneAI/pennylane/pull/3061)

   - `QueuingContext` is renamed to `QueuingManager`. It is no longer imported top level.
   - `QueuingManager` should now be the global communication point for putting queuable objects into the active queue.
   - `QueuingManager` is no longer an abstract base class.
   - `AnnotatedQueue` and its children no longer inherit from `QueuingManager`.
   - `QueuingManager` is no longer a context manager.
   -  Recording queues should start and stop recording via the `QueuingManager.add_active_queue` and 
     `QueueingContext.remove_active_queue` class methods instead of directly manipulating the `_active_contexts` property.
   - `AnnotatedQueue` and its children no longer provide global information about actively recording queues. This information
      is now only available through `QueuingManager`.
   - `AnnotatedQueue` and its children no longer have the private `_append`, `_remove`, `_update_info`, `_safe_update_info`,
      and `_get_info` methods. The public analogues should be used instead.
   

<h3>Breaking changes</h3>

 * `QueuingContext` is renamed `QueuingManager` and it is no longer updated top-level. The class may be accessed
  as `qml.queuing.QueuingManager`.
  [(#3061)](https://github.com/PennyLaneAI/pennylane/pull/3061)

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Christina Lee