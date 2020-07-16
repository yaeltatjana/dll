#!/bin/bash

# script to launch the benchmark tests

# script to be used from dll file
output_path="../benchmark/"


# to use from dll file
if [ ! -d ${output_path} ]
then
  mkdir ${output_path}
  echo "Create missing output directory "${output_path}
fi

# loop inspired by:
#     https://unix.stackexchange.com/questions/450944/bash-loop-through-list-of-strings?answertab=active#tab-top
# enter here which perf test you want to launch
perf=("dd" "ddd" "lenet" "alexnet" "vggnet16")
for t in "${perf[@]}"; do
    "./release/bin/${t}_perf"
done

