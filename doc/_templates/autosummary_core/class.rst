{% if referencefile %}
.. include:: {{ referencefile }}
{% endif %}

{% if module.split(".")[1:] | length >= 1 %}
	{% set mod = module.split(".")[1:] | join(".") %}
	{% set mod = "qml." + mod %}
{% else %}
	{% set mod = "qml" %}
{% endif %}

{{ mod }}.{{ objname }}
={% for i in range(mod|length) %}={% endfor %}{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% if '__init__' in methods %}
     {% set caught_result = methods.remove('__init__') %}
   {% endif %}

   .. raw:: html

      <h2>Class details</h2>

   {% block methods_summary %}
   {% if methods %}

   .. rubric:: Methods Summary

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% block attributes_documentation %}
   {% if attributes %}

   .. rubric:: Attributes Documentation

   {% for item in attributes %}
   .. autoattribute:: {{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% block methods_documentation %}
   {% if methods %}

   .. rubric:: Methods Documentation

   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}
