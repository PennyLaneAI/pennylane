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

      <a class="class-details-header" data-toggle="collapse" href="#classDetails" aria-expanded="false" aria-controls="classDetails">
         <h2>
            <i class="fas fa-angle-down rotate" style="float: right;"></i> Class details
         </h2>
      </a>
      <div class="collapse" id="classDetails">

   {% block attributes_documentation %}
   {% if attributes %}

   .. raw:: html

      <h3>Attributes documentation</h3>

   {% for item in attributes %}
   .. autoattribute:: {{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% block methods_documentation %}
   {% if methods %}

   .. raw:: html

      <h3>Methods documentation</h3>

   {% block methods_summary %}
   {% if methods %}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   .. raw:: html

      </div>
      <script type="text/javascript">
         $(".class-details-header").click(function () {
             $(this).children('h2').eq(0).children('i').eq(0).toggleClass("up");
         })
      </script>
