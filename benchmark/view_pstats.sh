#! /bin/bash
# View the contents of a pstats profile file using gprof2dot.

gprof2dot -f pstats "$1" | dot -Tpng -o output.png
eog output.png
