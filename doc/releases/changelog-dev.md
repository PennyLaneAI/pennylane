:orphan:

# Release 0.27.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

* Structural improvements are made to `QueuingContext` and `AnnotatedQueue`. None of these changes should 
  influence PennyLane behaviour outside of the `queueing.py` module.
  [(#2794)](https://github.com/PennyLaneAI/pennylane/pull/2794)

   - `QueuingContext` should now be the global communication point for putting queuable objects into the active queue.
   - `QueuingContext` is no longer an abstract base class.
   - `AnnotatedQueue` and its children no longer inherit from `QueuingContext`.
   - `QueuingContext` is no longer a context manager.
   -  Recording queues should start and stop recording via the `QueuingContext.add_active_queue` and 
     `QueueingContext.remove_active_queue` class methods instead of directly manipulating the `_active_contexts` property.
   - `AnnotatedQueue` and its children no longer provide global information about actively recording queues. This information
      is now only available through `QueuingContext`.
   - `AnnotatedQueue` and its children no longer have the private `_append`, `_remove`, `_update_info`, `_safe_update_info`,
      and `_get_info` methods. The public analogues should be used instead.
   

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

* `qml.tape.stop_recording` and `QuantumTape.stop_recording` are moved to `qml.queuing.QueuingManager.stop_recording`. The old functions will still be available
  untill v0.29.
  [(#3068)](https://github.com/PennyLaneAI/pennylane/pull/3068)

<h3>Documentation</h3>

<h3>Bug fixes</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Christina Lee