#!/bin/bash
missing_files=()
for file in $@
do
    if grep -Fxq $file tests/tests_passing_pylint.txt
    then
        :
    else
        missing_files+=($file)
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "All new test files are already added"
    exit 0
fi

echo "Please add the following test files to tests/tests_passing_pylint.txt:"
for file in ${missing_files[@]}; do echo $file; done
exit 1
