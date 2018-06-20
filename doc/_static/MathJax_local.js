document.write('<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>');
document.write('\
<script type="text/x-mathjax-config">\
          MathJax.Hub.Config({\
            TeX: {\
              Macros: {\
                expect: ["\\\\langle #1 \\\\rangle",1],\
                ket: ["\\\\left| \#1\\\\right\\\\rangle",1],\
                bra: ["\\\\left\\\\langle \#1\\\\right|",1],\
                braket: ["\\\\langle \#1 \\\\rangle",1],\
                braketD: ["\\\\langle \#1 \\\\mid \#2 \\\\rangle",2],\
                braketT: ["\\\\langle \#1 \\\\mid \#2 \\\\mid \#3 \\\\rangle",3],\
                ketbra: ["| #1 \\\\rangle \\\\langle #2 |",2],\
                hc: ["\\\\text{h.c.}",0],\
                cc: ["\\\\text{c.c.}",0],\
                pde: ["\\\\frac{\\\\partial}{\\\\partial \#1}",1],\
                R: ["\\\\mathbb{R}",0],\
                C: ["\\\\mathbb{C}",0],\
                I: ["I",0],\
                x: ["\\\\hat{x}",0],\
                p: ["\\\\hat{p}",0],\
                a: ["\\\\hat{a}",0],\
                ad: ["\\\\text{ad}",0],\
                Ad: ["\\\\text{Ad}",0],\
                Var: ["\\\\text{Var}",0],\
                re: ["\\\\text{Re}",0],\
                im: ["\\\\text{Im}",0],\
                tr: ["\\\\mathrm{Tr} #1",1],\
                sign: ["\\\\text{sign}",0]\
              }\
            }\
          });\
</script>')
