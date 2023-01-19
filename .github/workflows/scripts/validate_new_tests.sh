#!/bin/bash
missing_files=()
list_file=tests/tests_passing_pylint.txt

for file in $@
do
    # it's in the list, check the next one
    if grep -Fxq $file $list_file
    then
        continue
    fi

    # check if this file's folder is in the list
    filedir=$file
    while [ $filedir != "." ]
    do
        filedir=$(dirname $filedir)
        if grep -Fxq "${filedir}/**" $list_file
        then
            break
        fi
    done

    # loop reached termination condition without finding a match, user needs to add this file
    if [ ${filedir} == "." ]
    then
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
